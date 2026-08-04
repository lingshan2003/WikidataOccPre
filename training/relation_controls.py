"""Auditable relation-type controls for node-classification experiments.

The prepared graph stores an ID for each directed relation, including a
``__rev`` counterpart.  These helpers always operate on a base relation and
therefore apply a requested ablation to both directions.  They only mutate the
in-memory graph supplied to a training or explanation run; the source artifact
is never changed.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Set, Tuple

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


def apply_relation_controls(
    data,
    relation_ids_to_drop: Iterable[int] = (),
    shuffle_relation_types: bool = False,
    shuffle_seed: int | None = None,
) -> Dict[str, object]:
    """Remove selected directed relation IDs, then optionally permute types.

    A global permutation preserves the frequency of every remaining relation
    type and the graph topology exactly, while breaking their correspondence.
    The returned manifest contains enough information to replay the same
    transformation for attention export.
    """
    drop_ids = tuple(sorted({int(relation_id) for relation_id in relation_ids_to_drop}))
    edge_count_before = int(data.edge_type.numel())
    if drop_ids:
        drop_tensor = torch.tensor(drop_ids, dtype=data.edge_type.dtype, device=data.edge_type.device)
        keep = ~torch.isin(data.edge_type, drop_tensor)
        data.edge_index = data.edge_index[:, keep]
        data.edge_type = data.edge_type[keep]
    edge_count_after_ablation = int(data.edge_type.numel())

    if shuffle_relation_types:
        if shuffle_seed is None:
            raise ValueError("A shuffle seed is required when relation types are shuffled")
        generator = torch.Generator(device=data.edge_type.device)
        generator.manual_seed(int(shuffle_seed))
        permutation = torch.randperm(data.edge_type.numel(), generator=generator, device=data.edge_type.device)
        data.edge_type = data.edge_type[permutation]

    return {
        "dropped_relation_ids": list(drop_ids),
        "edge_count_before": edge_count_before,
        "edge_count_after_ablation": edge_count_after_ablation,
        "relation_type_shuffle": bool(shuffle_relation_types),
        "relation_type_shuffle_seed": int(shuffle_seed) if shuffle_relation_types else None,
    }
