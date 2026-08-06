"""Small utilities shared by the single-node and aggregate RGAT exporters."""

from typing import Mapping

import torch


def attention_relation_ids(layer_info: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Align returned RGAT attention edges with input relation IDs.

    ``RGATConv`` can remove existing self loops and append synthetic ones before
    returning ``edge_index`` and ``alpha``.  Those synthetic edges do not have a
    source relation in the prepared graph.  They receive ``-1`` here and callers
    should report or exclude them explicitly instead of treating them as a real
    relation.  Non-self-loop input edges retain their original ordering in the
    PyG tensor implementation; the key-based fallback keeps the export useful
    if a future PyG version changes that order.
    """
    attention_edge_index = layer_info["edge_index"]
    input_edge_index = layer_info.get("input_edge_index", attention_edge_index)
    input_edge_type = layer_info.get("input_edge_type", layer_info["edge_type"])
    aligned = torch.full(
        (attention_edge_index.size(1),),
        -1,
        dtype=input_edge_type.dtype,
        device=input_edge_type.device,
    )
    if attention_edge_index.size(1) == input_edge_type.numel() and torch.equal(
        attention_edge_index, input_edge_index
    ):
        return input_edge_type

    # PyG's remove_self_loops/add_self_loops path retains non-loop input edges
    # first, then appends its generated loops.
    non_loop = input_edge_index[0] != input_edge_index[1]
    retained_index = input_edge_index[:, non_loop]
    retained_type = input_edge_type[non_loop]
    retained_count = retained_type.numel()
    if retained_count and attention_edge_index.size(1) >= retained_count and torch.equal(
        attention_edge_index[:, :retained_count], retained_index
    ):
        aligned[:retained_count] = retained_type
        return aligned

    # Future-proof fallback.  Parallel edges with different relation IDs are
    # assigned in source order, which is the order used by current PyG layers.
    queues = {}
    for edge, relation_id in zip(retained_index.t().tolist(), retained_type.tolist()):
        queues.setdefault((int(edge[0]), int(edge[1])), []).append(int(relation_id))
    positions = {key: 0 for key in queues}
    for index, edge in enumerate(attention_edge_index.t().tolist()):
        key = (int(edge[0]), int(edge[1]))
        position = positions.get(key, 0)
        candidates = queues.get(key, ())
        if position < len(candidates):
            aligned[index] = candidates[position]
            positions[key] = position + 1
    return aligned
