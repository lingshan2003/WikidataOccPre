"""Auditable relation-type controls for node-classification experiments.

The prepared graph stores an ID for each directed relation, including a
``__rev`` counterpart.  These helpers always operate on a base relation and
therefore apply a requested ablation to both directions.  They only mutate the
in-memory graph supplied to a training or explanation run; the source artifact
is never changed.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch


# These groups are deliberately conservative and are based on the relation
# vocabulary in Q_R_Q_extended.txt.  Any relation not named below belongs to
# ``other`` rather than being silently assigned a questionable meaning.
RELATION_GROUPS: Mapping[str, frozenset[str]] = {
    "kinship": frozenset({
        "sibling", "child", "father", "mother", "spouse", "relative",
        "unmarried_partner", "significant_person", "godparent", "stepparent",
        "kinship_to_subject",
    }),
    "education_mentorship": frozenset({
        "student_of", "doctoral_advisor", "student", "doctoral_student", "trained_by", "head_coach",
    }),
    "professional_collaboration": frozenset({
        "employer", "partner_in_business_or_sport", "partnership_with", "co-driver",
        "copyright_representative", "director/manager", "owner_of", "owned_by", "sponsor",
    }),
    "influence_succession": frozenset({
        "influenced_by", "inspired_by", "interested_in", "follow", "followed_by",
        "replaces", "replaced_by",
    }),
    "religious": frozenset({"consecrator"}),
}


def parse_selection(value: str) -> Tuple[str, ...]:
    """Parse a comma-separated CLI selection, allowing ``none``."""
    if value.strip().casefold() == "none":
        return ()
    values = tuple(sorted({item.strip() for item in value.split(",") if item.strip()}))
    if not values:
        raise ValueError("Selection must be a comma-separated list or 'none'")
    return values


def base_relation_name(relation: str) -> str:
    """Return the original relation name for an explicitly generated reverse edge."""
    return relation.removesuffix("__rev")


def available_relation_groups(relation_to_id: Mapping[str, int]) -> Tuple[str, ...]:
    """Return groups that have at least one relation in this particular artifact."""
    bases = {base_relation_name(name) for name in relation_to_id}
    groups = sorted(name for name, members in RELATION_GROUPS.items() if members & bases)
    if bases - set().union(*RELATION_GROUPS.values()):
        groups.append("other")
    return tuple(groups)


def resolve_ablation(
    relation_to_id: Mapping[str, int],
    group_names: Sequence[str],
    relation_names: Sequence[str],
) -> Tuple[Set[int], Tuple[str, ...]]:
    """Resolve named relation groups and base relation names to directed IDs.

    ``other`` means every relation not covered by one of the explicit groups.
    Asking for an unavailable group or misspelled relation is an error, so an
    intended ablation never silently becomes a no-op.
    """
    base_names = {base_relation_name(name) for name in relation_to_id}
    known_groups = set(available_relation_groups(relation_to_id))
    unknown_groups = set(group_names) - known_groups
    if unknown_groups:
        raise ValueError(
            f"Unknown or unavailable relation groups: {sorted(unknown_groups)}. "
            f"Available: {sorted(known_groups)}"
        )
    unknown_relations = set(relation_names) - base_names
    if unknown_relations:
        raise ValueError(
            f"Unknown base relation names: {sorted(unknown_relations)}. "
            f"Use the original name without '__rev'."
        )

    explicitly_grouped = set().union(*RELATION_GROUPS.values())
    selected_bases = set(relation_names)
    for group_name in group_names:
        if group_name == "other":
            selected_bases.update(base_names - explicitly_grouped)
        else:
            selected_bases.update(RELATION_GROUPS[group_name] & base_names)

    relation_ids = {
        int(relation_id)
        for relation, relation_id in relation_to_id.items()
        if base_relation_name(relation) in selected_bases
    }
    return relation_ids, tuple(sorted(selected_bases))


def relation_pair_keys(data, relation_to_id: Mapping[str, int]) -> np.ndarray:
    """Encode each generated forward/reverse relation pair as one integer key.

    The key contains the base relation plus the unordered endpoint pair, so
    deleting a selected key always removes its two directed counterparts.  It
    also collapses duplicate source records that become identical after the
    preparation step's reverse-edge addition.
    """
    relation_ids = data.edge_type.detach().cpu().numpy().astype(np.int64, copy=False)
    source, target = data.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
    max_relation_id = max(relation_to_id.values())
    base_names = sorted({base_relation_name(name) for name in relation_to_id})
    base_to_id = {name: index for index, name in enumerate(base_names)}
    relation_to_base = np.full(max_relation_id + 1, -1, dtype=np.int64)
    for relation, relation_id in relation_to_id.items():
        relation_to_base[int(relation_id)] = base_to_id[base_relation_name(relation)]
    if relation_ids.size and (relation_ids.min() < 0 or relation_ids.max() >= len(relation_to_base)):
        raise ValueError("edge_type contains a relation ID absent from metadata")
    base_ids = relation_to_base[relation_ids]
    if (base_ids < 0).any():
        raise ValueError("edge_type contains a relation ID absent from metadata")
    low, high = np.minimum(source, target), np.maximum(source, target)
    node_count = int(data.num_nodes)
    return ((base_ids * node_count) + low) * node_count + high


def select_relation_pairs(
    data,
    relation_ids: Iterable[int],
    relation_to_id: Mapping[str, int],
    incident_node_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return selected base-relation/undirected pairs, optionally incident to nodes.

    The optional node selector is defined on original node indices.  Since a
    relation pair retains the same endpoint pair in both generated directions,
    selecting either endpoint preserves paired deletion of forward and reverse
    message edges.
    """
    selected = np.array(sorted({int(relation_id) for relation_id in relation_ids}), dtype=np.int64)
    if selected.size == 0:
        return np.empty(0, dtype=np.int64)
    edge_relation_ids = data.edge_type.detach().cpu().numpy()
    keep = np.isin(edge_relation_ids, selected)
    if incident_node_mask is not None:
        node_mask = np.asarray(incident_node_mask, dtype=bool)
        if node_mask.ndim != 1 or len(node_mask) != int(data.num_nodes):
            raise ValueError("incident_node_mask must be a Boolean vector with one entry per graph node")
        source, target = data.edge_index.detach().cpu().numpy().astype(np.int64, copy=False)
        keep &= node_mask[source] | node_mask[target]
    return np.unique(relation_pair_keys(data, relation_to_id)[keep])


def count_relation_pairs(
    data,
    relation_ids: Iterable[int],
    relation_to_id: Mapping[str, int],
    incident_node_mask: Optional[np.ndarray] = None,
) -> int:
    """Count relation pairs represented by selected types, optionally incident to nodes."""
    return int(len(select_relation_pairs(data, relation_ids, relation_to_id, incident_node_mask)))


def _drop_relation_pairs(data, relation_to_id: Mapping[str, int], pair_keys_to_drop: np.ndarray) -> int:
    """Delete whole relation pairs and return their actual unique count."""
    selected_keys = np.unique(np.asarray(pair_keys_to_drop, dtype=np.int64))
    if not len(selected_keys):
        return 0
    keys = relation_pair_keys(data, relation_to_id)
    present = np.isin(keys, selected_keys)
    if int(np.unique(keys[present]).size) != len(selected_keys):
        raise ValueError("Requested relation pair deletion contains pairs absent from this graph")
    keep_tensor = torch.from_numpy(~present).to(device=data.edge_type.device)
    data.edge_index = data.edge_index[:, keep_tensor]
    data.edge_type = data.edge_type[keep_tensor]
    return int(len(selected_keys))


def _drop_random_relation_pairs(
    data,
    relation_to_id: Mapping[str, int],
    pair_count: int,
    seed: int,
    candidate_pair_keys: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """Uniformly remove whole relation pairs while keeping generated reverses aligned."""
    if pair_count < 0:
        raise ValueError("random edge-pair drop count must be non-negative")
    keys = relation_pair_keys(data, relation_to_id)
    unique_keys = np.unique(keys)
    if candidate_pair_keys is not None:
        candidate_keys = np.unique(np.asarray(candidate_pair_keys, dtype=np.int64))
        candidate_keys = candidate_keys[np.isin(candidate_keys, unique_keys)]
        if len(candidate_keys) != len(np.unique(np.asarray(candidate_pair_keys, dtype=np.int64))):
            raise ValueError("Random-drop candidate relation pairs contain pairs absent from this graph")
    else:
        candidate_keys = unique_keys
    if pair_count > len(candidate_keys):
        raise ValueError(
            f"Cannot remove {pair_count} relation pairs: candidate set has only {len(candidate_keys)} pairs"
        )
    if pair_count == 0:
        return 0, int(len(candidate_keys))
    rng = np.random.default_rng(seed)
    dropped_keys = rng.choice(candidate_keys, size=pair_count, replace=False)
    _drop_relation_pairs(data, relation_to_id, dropped_keys)
    return int(pair_count), int(len(candidate_keys))


def apply_relation_controls(
    data,
    relation_ids_to_drop: Iterable[int] = (),
    relation_pair_keys_to_drop: Optional[np.ndarray] = None,
    relation_to_id: Mapping[str, int] | None = None,
    random_edge_drop_pairs: int = 0,
    random_edge_drop_seed: int | None = None,
    random_edge_drop_candidate_pair_keys: Optional[np.ndarray] = None,
    shuffle_relation_types: bool = False,
    shuffle_seed: int | None = None,
) -> Dict[str, object]:
    """Remove selected relation edges, then optionally random-drop or shuffle.

    A global permutation preserves the frequency of every remaining relation
    type and the graph topology exactly, while breaking their correspondence.
    The returned manifest contains enough information to replay the same
    transformation for attention export.
    """
    drop_ids = tuple(sorted({int(relation_id) for relation_id in relation_ids_to_drop}))
    if relation_pair_keys_to_drop is not None and drop_ids:
        raise ValueError("Use either relation IDs or relation-pair keys for an ablation, not both")
    if relation_pair_keys_to_drop is not None and relation_to_id is None:
        raise ValueError("relation_to_id metadata is required for relation-pair deletion")
    edge_count_before = int(data.edge_type.numel())
    if drop_ids:
        drop_tensor = torch.tensor(drop_ids, dtype=data.edge_type.dtype, device=data.edge_type.device)
        keep = ~torch.isin(data.edge_type, drop_tensor)
        data.edge_index = data.edge_index[:, keep]
        data.edge_type = data.edge_type[keep]
    dropped_relation_pair_count = _drop_relation_pairs(
        data, relation_to_id, relation_pair_keys_to_drop
    ) if relation_pair_keys_to_drop is not None else 0
    edge_count_after_ablation = int(data.edge_type.numel())

    if random_edge_drop_pairs and relation_to_id is None:
        raise ValueError("relation_to_id metadata is required for random edge-pair deletion")
    if random_edge_drop_pairs and random_edge_drop_seed is None:
        raise ValueError("A random-drop seed is required when deleting random relation pairs")
    dropped_random_pairs, available_relation_pairs = _drop_random_relation_pairs(
        data,
        relation_to_id,
        int(random_edge_drop_pairs),
        int(random_edge_drop_seed),
        random_edge_drop_candidate_pair_keys,
    ) if random_edge_drop_pairs else (0, None)
    edge_count_after_random_drop = int(data.edge_type.numel())

    if shuffle_relation_types:
        if shuffle_seed is None:
            raise ValueError("A shuffle seed is required when relation types are shuffled")
        generator = torch.Generator(device=data.edge_type.device)
        generator.manual_seed(int(shuffle_seed))
        permutation = torch.randperm(data.edge_type.numel(), generator=generator, device=data.edge_type.device)
        data.edge_type = data.edge_type[permutation]

    return {
        "dropped_relation_ids": list(drop_ids),
        "dropped_relation_pair_count": dropped_relation_pair_count,
        "edge_count_before": edge_count_before,
        "edge_count_after_ablation": edge_count_after_ablation,
        "random_edge_drop_pairs": dropped_random_pairs,
        "random_edge_drop_available_pairs": available_relation_pairs,
        "random_edge_drop_seed": int(random_edge_drop_seed) if random_edge_drop_pairs else None,
        "edge_count_after_random_drop": edge_count_after_random_drop,
        "relation_type_shuffle": bool(shuffle_relation_types),
        "relation_type_shuffle_seed": int(shuffle_seed) if shuffle_relation_types else None,
    }
