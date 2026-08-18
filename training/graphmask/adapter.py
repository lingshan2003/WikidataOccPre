"""Message capture and replacement adapters for the three classifier families."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from models import build_feature_specs, build_model

from .core import LayerTrace


def _replace_message(
    message: torch.Tensor, gate: torch.Tensor, baseline: torch.Tensor
) -> torch.Tensor:
    if message.size(0) != gate.numel():
        raise RuntimeError(
            f"GraphMask gate has {gate.numel()} edges but the layer emitted {message.size(0)} messages"
        )
    if tuple(message.shape[1:]) != tuple(baseline.shape):
        raise RuntimeError(
            "GraphMask baseline shape does not match the layer message: "
            f"{tuple(baseline.shape)} != {tuple(message.shape[1:])}"
        )
    broadcast_gate = gate.reshape(gate.size(0), *([1] * (message.dim() - 1)))
    broadcast_baseline = baseline.reshape(1, *baseline.shape)
    return broadcast_gate * message + (1.0 - broadcast_gate) * broadcast_baseline


@dataclass(frozen=True)
class RestoredGraphMaskModel:
    adapter: "GraphMaskModelAdapter"
    feature_schema: Mapping[str, object]
    metadata: Mapping[str, object]
    model_name: str


class GraphMaskModelAdapter:
    """Expose original traces and differentiably replace per-edge messages."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        reference_model: nn.Module | None = None,
        equivalence_tolerance: float = 1e-5,
    ) -> None:
        if model_name not in {"rgat", "rgcn", "compgcn"}:
            raise ValueError(f"GraphMask does not support model {model_name!r}")
        self.model = model.eval()
        self.reference_model = (reference_model or model).eval()
        self.model_name = model_name
        self.equivalence_tolerance = float(equivalence_tolerance)
        self._equivalence_checked = self.reference_model is self.model
        for module in {self.model, self.reference_model}:
            for parameter in module.parameters():
                parameter.requires_grad = False

    @property
    def layers(self) -> Sequence[nn.Module]:
        return self.model.convs

    def _message_module(self, layer: nn.Module) -> nn.Module:
        return layer.message if self.model_name == "compgcn" else layer

    @contextmanager
    def _capture_hooks(self) -> Iterator[tuple[list[dict[str, torch.Tensor]], list[torch.Tensor]]]:
        layer_inputs: list[dict[str, torch.Tensor]] = [{} for _ in self.layers]
        messages: list[torch.Tensor | None] = [None for _ in self.layers]
        handles = []

        for layer_index, layer in enumerate(self.layers):
            def pre_hook(module, args, kwargs, index=layer_index):
                del module, kwargs
                if self.model_name == "compgcn":
                    node_state, _, edge_index, edge_type = args[:4]
                else:
                    node_state, edge_index, edge_type = args[:3]
                layer_inputs[index] = {
                    "node_state": node_state,
                    "edge_index": edge_index,
                    "edge_type": edge_type,
                }

            def message_hook(module, args, output, index=layer_index):
                del module, args
                if messages[index] is not None:
                    raise RuntimeError(f"GraphMask layer {index} emitted messages more than once")
                messages[index] = output
                return output

            handles.append(layer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            message_module = self._message_module(layer)
            if self.model_name == "compgcn":
                handles.append(message_module.register_forward_hook(message_hook))
            else:
                handles.append(message_module.register_message_forward_hook(message_hook))
        try:
            yield layer_inputs, messages  # type: ignore[arg-type]
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def _replacement_hooks(
        self,
        gates: Sequence[torch.Tensor],
        baselines: Sequence[torch.Tensor],
    ) -> Iterator[None]:
        if len(gates) != len(self.layers) or len(baselines) != len(self.layers):
            raise ValueError("GraphMask replacement count does not match the model depth")
        handles = []
        for layer_index, layer in enumerate(self.layers):
            def replacement_hook(module, args, output, index=layer_index):
                del module, args
                return _replace_message(output, gates[index], baselines[index])

            message_module = self._message_module(layer)
            if self.model_name == "compgcn":
                handles.append(message_module.register_forward_hook(replacement_hook))
            else:
                handles.append(message_module.register_message_forward_hook(replacement_hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _forward(self, model: nn.Module, features, edge_index, edge_type) -> torch.Tensor:
        output = model(features, edge_index, edge_type)
        if isinstance(output, tuple):
            return output[0]
        return output

    @torch.no_grad()
    def _check_equivalence(self, features, edge_index, edge_type) -> torch.Tensor:
        reference_logits = self._forward(
            self.reference_model, features, edge_index, edge_type
        )
        candidate_logits = self._forward(self.model, features, edge_index, edge_type)
        maximum_error = float((reference_logits - candidate_logits).abs().max().item())
        if maximum_error > self.equivalence_tolerance:
            raise RuntimeError(
                "The FastRGCNConv GraphMask copy is not numerically equivalent to the source "
                f"checkpoint (max |delta logits|={maximum_error:.3g}, allowed "
                f"{self.equivalence_tolerance:.3g})"
            )
        self._equivalence_checked = True
        return reference_logits

    @torch.no_grad()
    def trace(self, features, edge_index, edge_type) -> tuple[torch.Tensor, list[LayerTrace]]:
        if not self._equivalence_checked:
            original_logits = self._check_equivalence(features, edge_index, edge_type)
        else:
            original_logits = None
        with self._capture_hooks() as (layer_inputs, messages):
            traced_logits = self._forward(self.model, features, edge_index, edge_type)
        if original_logits is None:
            original_logits = traced_logits
        traces = []
        for layer_index, (inputs, message) in enumerate(zip(layer_inputs, messages)):
            if not inputs or message is None:
                raise RuntimeError(f"GraphMask did not observe messages from layer {layer_index}")
            observed_edges = inputs["edge_index"]
            if observed_edges.size(1) != message.size(0):
                raise RuntimeError(
                    f"Layer {layer_index} changed edge order/count inside message passing; "
                    "this PyG configuration cannot be masked safely"
                )
            node_state = inputs["node_state"]
            traces.append(LayerTrace(
                source_state=node_state[observed_edges[0]].detach(),
                target_state=node_state[observed_edges[1]].detach(),
                message=message.detach(),
                edge_index=observed_edges.detach(),
                edge_type=inputs["edge_type"].detach(),
            ))
        return original_logits.detach(), traces

    def masked_forward(
        self,
        features,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        gates: Sequence[torch.Tensor],
        baselines: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        with self._replacement_hooks(gates, baselines):
            return self._forward(self.model, features, edge_index, edge_type)


def restore_graphmask_model(
    checkpoint: Mapping[str, object], device: torch.device
) -> RestoredGraphMaskModel:
    metadata = checkpoint["metadata"]
    model_name = str(checkpoint.get("model_name", "rgat"))
    feature_schema = checkpoint.get("model_feature_schema", metadata["feature_schema"])
    model_config = dict(checkpoint.get("model_config", {}))
    common = {
        "num_relations": metadata["num_relations"],
        "num_classes": metadata["num_classes"],
        "feature_specs": build_feature_specs(feature_schema, metadata),
    }
    reference_model = build_model(model_name, **common, **model_config).to(device)
    reference_model.load_state_dict(checkpoint["state_dict"])

    if model_name == "rgcn" and model_config.get("rgcn_backend", "fast") != "fast":
        fast_config = dict(model_config)
        fast_config["rgcn_backend"] = "fast"
        model = build_model(model_name, **common, **fast_config).to(device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model = reference_model
    adapter = GraphMaskModelAdapter(
        model, model_name=model_name, reference_model=reference_model
    )
    return RestoredGraphMaskModel(adapter, feature_schema, metadata, model_name)
