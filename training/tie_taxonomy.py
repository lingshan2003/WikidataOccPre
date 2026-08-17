"""Versioned inherited/acquired relationship taxonomies.

The graph contains directed relation labels because preparation adds a
``__rev`` edge for every source relation.  Taxonomies deliberately operate on
the original base relation names so every intervention, audit, and explanation
assigns both message directions to the same social-tie category.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence, Set, Tuple

DEFAULT_TIE_TAXONOMY_PATH = Path(__file__).resolve().parents[1] / "config" / "tie_taxonomy_v1.json"
TIE_GROUPS = ("inherited", "acquired")


def base_relation_name(relation: str) -> str:
    """Return the base name for a generated reverse edge without importing PyTorch controls."""
    return relation.removesuffix("__rev")


@dataclass(frozen=True)
class TieTaxonomy:
    """A fully resolved, artifact-specific two-way relation classification."""

    name: str
    version: int
    path: Path
    sha256: str
    inherited: Tuple[str, ...]
    acquired: Tuple[str, ...]

    @property
    def groups(self) -> Mapping[str, Tuple[str, ...]]:
        return {"inherited": self.inherited, "acquired": self.acquired}

    def group_for_base_relation(self, relation: str) -> str:
        base_relation = base_relation_name(relation)
        for group, members in self.groups.items():
            if base_relation in members:
                return group
        raise ValueError(
            f"Relation {relation!r} is not covered by taxonomy {self.name!r}; "
            "load_tie_taxonomy should have rejected this configuration."
        )

    def manifest(self) -> Dict[str, object]:
        """Return JSON-safe provenance persisted by runs and reports."""
        return {
            "name": self.name,
            "version": self.version,
            "path": str(self.path),
            "sha256": self.sha256,
            "groups": {group: list(members) for group, members in self.groups.items()},
        }


@dataclass(frozen=True)
class RelationTaxonomy:
    """A fully resolved, artifact-specific taxonomy with any number of groups.

    ``TieTaxonomy`` above remains the compatibility contract for the original
    inherited/acquired audit.  This generic form is intentionally separate so
    a finer taxonomy cannot accidentally be consumed as the old two-way
    taxonomy by an existing report.
    """

    name: str
    version: int
    path: Path
    sha256: str
    group_members: Mapping[str, Tuple[str, ...]]

    @property
    def groups(self) -> Mapping[str, Tuple[str, ...]]:
        return self.group_members

    def group_for_base_relation(self, relation: str) -> str:
        base_relation = base_relation_name(relation)
        for group, members in self.groups.items():
            if base_relation in members:
                return group
        raise ValueError(
            f"Relation {relation!r} is not covered by taxonomy {self.name!r}; "
            "load_relation_taxonomy should have rejected this configuration."
        )

    def manifest(self) -> Dict[str, object]:
        """Return JSON-safe provenance persisted by subgroup audit runs."""
        return {
            "name": self.name,
            "version": self.version,
            "path": str(self.path),
            "sha256": self.sha256,
            "groups": {group: list(members) for group, members in self.groups.items()},
        }


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a config or data artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_relations(relation_to_id: Mapping[str, int]) -> Set[str]:
    if not relation_to_id:
        raise ValueError("A relation taxonomy requires a non-empty relation_to_id mapping")
    return {base_relation_name(str(relation)) for relation in relation_to_id}


def _parse_explicit_group(value: object, group: str) -> Set[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"taxonomy groups.{group} must be a list of base relation names")
    relations = [item.strip() for item in value]
    if any(not item for item in relations):
        raise ValueError(f"taxonomy groups.{group} may not contain empty relation names")
    if any(item.endswith("__rev") for item in relations):
        raise ValueError(
            f"taxonomy groups.{group} must use base relation names, not generated '__rev' labels"
        )
    if len(set(relations)) != len(relations):
        raise ValueError(f"taxonomy groups.{group} contains duplicate relation names")
    return set(relations)


def _load_taxonomy_payload(path: str | Path) -> Tuple[Path, Mapping[str, object]]:
    """Read the common JSON envelope used by two-way and subgroup taxonomies."""
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Relation taxonomy does not exist: {resolved_path}")
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Relation taxonomy is not valid JSON: {resolved_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Relation taxonomy must be a JSON object")
    return resolved_path, payload


def load_relation_taxonomy(
    path: str | Path,
    relation_to_id: Mapping[str, int],
) -> RelationTaxonomy:
    """Load a complete, versioned multi-group taxonomy for an artifact.

    Every base relation must belong to exactly one named group.  One group may
    use ``\"all_remaining\"`` as an explicit residual bucket; it resolves only
    after all other groups have been validated against the artifact vocabulary.
    """
    resolved_path, payload = _load_taxonomy_payload(path)
    name = payload.get("name")
    version = payload.get("version")
    groups = payload.get("groups")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Relation taxonomy requires a non-empty string 'name'")
    if not isinstance(version, int) or version < 1:
        raise ValueError("Relation taxonomy requires an integer 'version' of at least one")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Relation taxonomy requires a non-empty object 'groups'")
    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("Relation taxonomy group names must be non-empty strings")
    normalized_groups = {str(group).strip(): value for group, value in groups.items()}
    if len(normalized_groups) != len(groups):
        raise ValueError("Relation taxonomy contains duplicate group names after trimming")

    available = _base_relations(relation_to_id)
    resolved_groups: Dict[str, Set[str]] = {}
    residual_groups = []
    for group, value in normalized_groups.items():
        if value == "all_remaining":
            residual_groups.append(group)
            continue
        members = _parse_explicit_group(value, group)
        unknown = members - available
        if unknown:
            raise ValueError(
                f"Relation taxonomy group {group!r} references relations absent from this artifact: "
                f"{sorted(unknown)}"
            )
        resolved_groups[group] = members
    if len(residual_groups) > 1:
        raise ValueError("Only one relation taxonomy group may use 'all_remaining'")

    explicit_members = set().union(*resolved_groups.values()) if resolved_groups else set()
    if len(explicit_members) != sum(len(members) for members in resolved_groups.values()):
        overlap = sorted(
            relation
            for relation in explicit_members
            if sum(relation in members for members in resolved_groups.values()) > 1
        )
        raise ValueError(f"Relation taxonomy groups overlap: {overlap}")
    if residual_groups:
        resolved_groups[residual_groups[0]] = available - explicit_members
    covered = set().union(*resolved_groups.values()) if resolved_groups else set()
    if covered != available:
        missing = sorted(available - covered)
        extra = sorted(covered - available)
        raise ValueError(
            "Relation taxonomy must cover every base relation exactly once; "
            f"missing={missing}, extra={extra}"
        )
    return RelationTaxonomy(
        name=name.strip(),
        version=version,
        path=resolved_path,
        sha256=sha256_file(resolved_path),
        group_members={group: tuple(sorted(members)) for group, members in sorted(resolved_groups.items())},
    )


def load_tie_taxonomy(
    path: str | Path | None,
    relation_to_id: Mapping[str, int],
) -> TieTaxonomy:
    """Load, validate, and resolve a taxonomy against an artifact vocabulary.

    ``acquired`` can be the literal string ``"all_remaining"``.  This is the
    default because it keeps the audit complete when a new source export adds
    a non-blood relation.  An explicit acquired list is also supported, but
    must cover every remaining base relation exactly.
    """
    resolved_path = Path(path) if path is not None else DEFAULT_TIE_TAXONOMY_PATH
    _, payload = _load_taxonomy_payload(resolved_path)
    groups = payload.get("groups")
    if not isinstance(groups, dict) or set(groups) != set(TIE_GROUPS):
        raise ValueError("Tie taxonomy groups must contain exactly 'inherited' and 'acquired'")
    generic = load_relation_taxonomy(resolved_path, relation_to_id)
    if not generic.groups["inherited"] or not generic.groups["acquired"]:
        raise ValueError("Tie taxonomy inherited and acquired groups must both be non-empty for this artifact")
    return TieTaxonomy(
        name=generic.name,
        version=generic.version,
        path=generic.path,
        sha256=generic.sha256,
        inherited=generic.groups["inherited"],
        acquired=generic.groups["acquired"],
    )


def parse_tie_group_selection(value: str) -> Tuple[str, ...]:
    """Parse inherited/acquired selections, allowing ``none`` like legacy controls."""
    stripped = value.strip().casefold()
    if stripped == "none":
        return ()
    selected = tuple(sorted({item.strip() for item in value.split(",") if item.strip()}))
    if not selected:
        raise ValueError("Tie-group selection must be 'none', 'inherited', 'acquired', or both")
    unknown = set(selected) - set(TIE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown tie groups: {sorted(unknown)}. Available: {list(TIE_GROUPS)}")
    return selected


def parse_relation_taxonomy_group_selection(value: str) -> Tuple[str, ...]:
    """Parse a generic taxonomy group selection, deferring membership checks.

    The selected taxonomy is artifact-specific, so the caller validates group
    names through :func:`resolve_relation_taxonomy_ablation` after loading it.
    """
    stripped = value.strip().casefold()
    if stripped == "none":
        return ()
    selected = tuple(sorted({item.strip() for item in value.split(",") if item.strip()}))
    if not selected:
        raise ValueError("Relation-taxonomy selection must be a group name, comma-separated names, or 'none'")
    return selected


def resolve_relation_taxonomy_ablation(
    taxonomy: RelationTaxonomy,
    group_names: Sequence[str],
    relation_to_id: Mapping[str, int],
) -> Tuple[Set[int], Tuple[str, ...]]:
    """Resolve arbitrary taxonomy groups to both directed relation IDs."""
    selected = set(group_names)
    unknown = selected - set(taxonomy.groups)
    if unknown:
        raise ValueError(
            f"Unknown relation taxonomy groups: {sorted(unknown)}. "
            f"Available: {sorted(taxonomy.groups)}"
        )
    selected_bases = set()
    for group in selected:
        selected_bases.update(taxonomy.groups[group])
    relation_ids = {
        int(relation_id)
        for relation, relation_id in relation_to_id.items()
        if base_relation_name(str(relation)) in selected_bases
    }
    return relation_ids, tuple(sorted(selected_bases))


def resolve_tie_ablation(
    taxonomy: TieTaxonomy,
    group_names: Sequence[str],
    relation_to_id: Mapping[str, int],
) -> Tuple[Set[int], Tuple[str, ...]]:
    """Resolve selected categories to all matching directed relation IDs."""
    selected = set(group_names)
    unknown = selected - set(TIE_GROUPS)
    if unknown:
        raise ValueError(f"Unknown tie groups: {sorted(unknown)}. Available: {list(TIE_GROUPS)}")
    selected_bases = set()
    for group in selected:
        selected_bases.update(taxonomy.groups[group])
    relation_ids = {
        int(relation_id)
        for relation, relation_id in relation_to_id.items()
        if base_relation_name(str(relation)) in selected_bases
    }
    return relation_ids, tuple(sorted(selected_bases))
