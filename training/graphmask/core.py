"""Modernized GraphMask probe components.

The design follows MichSchli/GraphMask (MIT): a layer-local classifier predicts
Hard-Concrete message gates, learned baselines replace closed messages, and a
Lagrange multiplier balances expected L0 sparsity against output fidelity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LayerTrace:
    """Original-model inputs and messages for one message-passing layer."""

    source_state: torch.Tensor
    target_state: torch.Tensor
    message: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor


class HardConcrete(nn.Module):
    """Hard-Concrete stochastic gate with an expected non-zero probability."""

    def __init__(
        self,
        temperature: float = 1.0 / 3.0,
        lower: float = -0.2,
        upper: float = 1.0,
        location_bias: float = 3.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not lower < 0 < upper:
            raise ValueError("Hard-Concrete bounds must satisfy lower < 0 < upper")
        self.temperature = float(temperature)
        self.lower = float(lower)
        self.upper = float(upper)
        self.location_bias = float(location_bias)
        self._log_ratio = math.log(-self.lower / self.upper)

    def expected_nonzero(self, logits: torch.Tensor) -> torch.Tensor:
        shifted = logits + self.location_bias
        return torch.sigmoid(shifted - self.temperature * self._log_ratio)

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shifted = logits + self.location_bias
        if self.training:
            uniform = torch.empty_like(shifted).uniform_(1e-6, 1.0 - 1e-6)
            concrete = torch.sigmoid(
                (torch.log(uniform) - torch.log1p(-uniform) + shifted) / self.temperature
            )
        else:
            concrete = torch.sigmoid(shifted)
        stretched = concrete * (self.upper - self.lower) + self.lower
        clipped = stretched.clamp(0.0, 1.0)
        hard = (clipped > 0.5).to(clipped.dtype)
        # Straight-through hard decisions: forward is binary, gradients follow
        # the clipped continuous relaxation.
        gate = clipped + (hard - clipped).detach()
        return gate, self.expected_nonzero(logits)


class MultiInputProjection(nn.Module):
    """Project and separately normalize heterogeneous gate inputs."""

    def __init__(self, input_dims: Sequence[int], output_dim: int) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(
            nn.Linear(int(input_dim), output_dim, bias=False) for input_dim in input_dims
        )
        self.norms = nn.ModuleList(nn.LayerNorm(output_dim) for _ in input_dims)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        fan_in = sum(int(value) for value in input_dims)
        bound = math.sqrt(6.0 / float(fan_in + output_dim))
        for transform in self.transforms:
            nn.init.uniform_(transform.weight, -bound, bound)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != len(self.transforms):
            raise ValueError("Gate projection received the wrong number of inputs")
        output = self.bias
        for tensor, transform, norm in zip(inputs, self.transforms, self.norms):
            output = output + norm(transform(tensor))
        return output / len(inputs)


class GraphMaskGate(nn.Module):
    def __init__(
        self,
        source_dim: int,
        message_dim: int,
        target_dim: int,
        hidden_dim: int,
        temperature: float,
        location_bias: float,
    ) -> None:
        super().__init__()
        self.source_dim = int(source_dim)
        self.message_dim = int(message_dim)
        self.target_dim = int(target_dim)
        self.projection = MultiInputProjection(
            [source_dim, message_dim, target_dim], hidden_dim
        )
        self.output = nn.Linear(hidden_dim, 1)
        self.hard_concrete = HardConcrete(
            temperature=temperature, location_bias=location_bias
        )

    def forward(self, trace: LayerTrace) -> tuple[torch.Tensor, torch.Tensor]:
        source = trace.source_state.detach().reshape(trace.source_state.size(0), self.source_dim)
        message = trace.message.detach().reshape(trace.message.size(0), self.message_dim)
        target = trace.target_state.detach().reshape(trace.target_state.size(0), self.target_dim)
        hidden = torch.relu(self.projection([source, message, target]))
        logits = self.output(hidden).squeeze(-1)
        return self.hard_concrete(logits)


class GraphMaskProbe(nn.Module):
    """One amortized message gate and learned replacement per GNN layer."""

    def __init__(
        self,
        layer_specs: Sequence[Mapping[str, object]],
        temperature: float = 1.0 / 3.0,
        location_bias: float = 3.0,
    ) -> None:
        super().__init__()
        if not layer_specs:
            raise ValueError("GraphMask requires at least one message-passing layer")
        normalized_specs = []
        gates = []
        baselines = []
        for spec in layer_specs:
            message_shape = tuple(int(value) for value in spec["message_shape"])
            source_dim = int(spec["source_dim"])
            target_dim = int(spec["target_dim"])
            message_dim = math.prod(message_shape)
            hidden_dim = int(spec.get("hidden_dim", target_dim))
            normalized_specs.append({
                "source_dim": source_dim,
                "target_dim": target_dim,
                "message_shape": list(message_shape),
                "hidden_dim": hidden_dim,
            })
            gates.append(GraphMaskGate(
                source_dim,
                message_dim,
                target_dim,
                hidden_dim,
                temperature,
                location_bias,
            ))
            baseline = torch.empty(message_shape)
            baseline.uniform_(-1.0 / math.sqrt(message_dim), 1.0 / math.sqrt(message_dim))
            baselines.append(nn.Parameter(baseline))
        self.layer_specs = normalized_specs
        self.temperature = float(temperature)
        self.location_bias = float(location_bias)
        self.gates = nn.ModuleList(gates)
        self.baselines = nn.ParameterList(baselines)
        self.register_buffer("enabled_layers", torch.zeros(len(gates), dtype=torch.bool))
        for parameter in self.parameters():
            parameter.requires_grad = False

    @classmethod
    def from_traces(
        cls,
        traces: Sequence[LayerTrace],
        temperature: float = 1.0 / 3.0,
        location_bias: float = 3.0,
    ) -> "GraphMaskProbe":
        specs = []
        for trace in traces:
            specs.append({
                "source_dim": math.prod(trace.source_state.shape[1:]),
                "target_dim": math.prod(trace.target_state.shape[1:]),
                "message_shape": list(trace.message.shape[1:]),
                "hidden_dim": math.prod(trace.target_state.shape[1:]),
            })
        return cls(specs, temperature=temperature, location_bias=location_bias)

    def enable_layer(self, layer: int) -> None:
        if layer < 0 or layer >= len(self.gates):
            raise IndexError(f"GraphMask layer {layer} is out of range")
        self.enabled_layers[layer] = True
        for parameter in self.gates[layer].parameters():
            parameter.requires_grad = True
        self.baselines[layer].requires_grad = True

    def enable_recorded_layers(self) -> None:
        enabled = self.enabled_layers.detach().cpu().tolist()
        for layer, value in enumerate(enabled):
            if value:
                self.enable_layer(layer)

    def forward(
        self, traces: Sequence[LayerTrace]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, int]:
        if len(traces) != len(self.gates):
            raise ValueError(
                f"Probe has {len(self.gates)} layers but adapter returned {len(traces)} traces"
            )
        gates: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []
        expected_count = traces[0].message.new_zeros(())
        eligible_count = 0
        for layer, (gate_module, trace) in enumerate(zip(self.gates, traces)):
            if bool(self.enabled_layers[layer]):
                gate, probability = gate_module(trace)
                expected_count = expected_count + probability.sum()
                eligible_count += int(probability.numel())
            else:
                gate = torch.ones(
                    trace.message.size(0), dtype=trace.message.dtype, device=trace.message.device
                )
                probability = torch.ones_like(gate)
            gates.append(gate)
            probabilities.append(probability)
        return gates, probabilities, expected_count, eligible_count

    def config(self) -> dict[str, object]:
        return {
            "layer_specs": self.layer_specs,
            "temperature": self.temperature,
            "location_bias": self.location_bias,
        }


class LagrangianOptimization:
    """Minimize f + lambda*g while maximizing the non-negative lambda."""

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        learning_rate: float,
        multiplier_learning_rate: float,
        device: torch.device,
        initial_alpha: float = 0.55,
        minimum_alpha: float = -2.0,
        maximum_alpha: float = 30.0,
    ) -> None:
        self.minimum_alpha = float(minimum_alpha)
        self.maximum_alpha = float(maximum_alpha)
        self.optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        self.alpha = torch.tensor(initial_alpha, device=device, requires_grad=True)
        self.multiplier_optimizer = torch.optim.RMSprop(
            [self.alpha], lr=multiplier_learning_rate, centered=True
        )

    @property
    def multiplier(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.alpha)

    def update(self, objective: torch.Tensor, constraint: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        self.multiplier_optimizer.zero_grad()
        loss = objective + self.multiplier * constraint
        loss.backward()
        self.optimizer.step()
        if self.alpha.grad is not None:
            self.alpha.grad.mul_(-1.0)
        self.multiplier_optimizer.step()
        with torch.no_grad():
            self.alpha.clamp_(self.minimum_alpha, self.maximum_alpha)
        return float(loss.detach().item())
