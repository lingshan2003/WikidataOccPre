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
    resolved_path = resolved_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Tie taxonomy does not exist: {resolved_path}")
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Tie taxonomy is not valid JSON: {resolved_path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Tie taxonomy must be a JSON object")
    name = payload.get("name")
    version = payload.get("version")
    groups = payload.get("groups")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Tie taxonomy requires a non-empty string 'name'")
    if not isinstance(version, int) or version < 1:
        raise ValueError("Tie taxonomy requires an integer 'version' of at least one")
    if not isinstance(groups, dict) or set(groups) != set(TIE_GROUPS):
        raise ValueError("Tie taxonomy groups must contain exactly 'inherited' and 'acquired'")

    available = _base_relations(relation_to_id)
    inherited = _parse_explicit_group(groups["inherited"], "inherited")
    unknown_inherited = inherited - available
    if unknown_inherited:
        raise ValueError(
            f"Tie taxonomy inherited group references relations absent from this artifact: {sorted(unknown_inherited)}"
        )

    acquired_value = groups["acquired"]
    if acquired_value == "all_remaining":
        acquired = available - inherited
    else:
        acquired = _parse_explicit_group(acquired_value, "acquired")
        unknown_acquired = acquired - available
        if unknown_acquired:
            raise ValueError(
                f"Tie taxonomy acquired group references relations absent from this artifact: {sorted(unknown_acquired)}"
            )
    overlap = inherited & acquired
    if overlap:
        raise ValueError(f"Tie taxonomy groups overlap: {sorted(overlap)}")
    covered = inherited | acquired
    if covered != available:
        missing = sorted(available - covered)
        extra = sorted(covered - available)
        raise ValueError(
            "Tie taxonomy must cover every base relation exactly once; "
            f"missing={missing}, extra={extra}"
        )
    if not inherited or not acquired:
        raise ValueError("Tie taxonomy inherited and acquired groups must both be non-empty for this artifact")
    return TieTaxonomy(
        name=name.strip(),
        version=version,
        path=resolved_path,
        sha256=sha256_file(resolved_path),
        inherited=tuple(sorted(inherited)),
        acquired=tuple(sorted(acquired)),
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
