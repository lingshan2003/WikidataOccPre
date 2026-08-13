"""Versioned, editable birth-cohort definitions for time-stratified audits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_BIRTH_COHORT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "birth_cohorts_historical_eras_v1.json"
)
MISSING_COHORT_ID = "missing_birth_year"
MISSING_COHORT_LABEL = "Missing birth year"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BirthCohort:
    identifier: str
    label: str
    start: Optional[int]
    end: Optional[int]

    def contains(self, years: np.ndarray) -> np.ndarray:
        result = np.isfinite(years)
        if self.start is not None:
            result &= years >= self.start
        if self.end is not None:
            result &= years <= self.end
        return result

    def manifest(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"id": self.identifier, "label": self.label}
        if self.start is not None:
            payload["start"] = self.start
        if self.end is not None:
            payload["end"] = self.end
        return payload


@dataclass(frozen=True)
class BirthCohortConfig:
    name: str
    version: int
    path: Path
    sha256: str
    time_field: str
    missing_policy: str
    cohorts: Tuple[BirthCohort, ...]

    @property
    def identifiers(self) -> Tuple[str, ...]:
        return tuple(cohort.identifier for cohort in self.cohorts)

    def cohort(self, identifier: str) -> BirthCohort:
        for cohort in self.cohorts:
            if cohort.identifier == identifier:
                return cohort
        raise ValueError(
            f"Unknown birth cohort {identifier!r}. Available: {list(self.identifiers)}"
        )

    def manifest(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "path": str(self.path),
            "sha256": self.sha256,
            "time_field": self.time_field,
            "missing_policy": self.missing_policy,
            "bins": [cohort.manifest() for cohort in self.cohorts],
        }


def _optional_integer(value: object, field: str, cohort_id: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Cohort {cohort_id!r} field {field!r} must be an integer when provided")
    return value


def _parse_cohort(value: object) -> BirthCohort:
    if not isinstance(value, Mapping):
        raise ValueError("Every birth cohort bin must be a JSON object")
    identifier, label = value.get("id"), value.get("label")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Every birth cohort bin requires a non-empty string 'id'")
    if identifier == MISSING_COHORT_ID:
        raise ValueError(f"{MISSING_COHORT_ID!r} is reserved for transparently reported missing dates")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"Cohort {identifier!r} requires a non-empty string 'label'")
    start = _optional_integer(value.get("start"), "start", identifier)
    end = _optional_integer(value.get("end"), "end", identifier)
    if start is not None and end is not None and start > end:
        raise ValueError(f"Cohort {identifier!r} has start greater than end")
    return BirthCohort(identifier.strip(), label.strip(), start, end)


def _validate_exhaustive_ordered_bins(cohorts: Sequence[BirthCohort]) -> None:
    if not cohorts:
        raise ValueError("Birth cohort configuration requires at least one bin")
    if cohorts[0].start is not None:
        raise ValueError("The first birth cohort bin must have no 'start' to cover early valid years")
    if cohorts[-1].end is not None:
        raise ValueError("The final birth cohort bin must have no 'end' to cover recent valid years")
    for previous, current in zip(cohorts, cohorts[1:]):
        if previous.end is None or current.start is None:
            raise ValueError("Only the first/last birth cohort bins may have open bounds")
        if current.start != previous.end + 1:
            raise ValueError(
                "Birth cohort bins must be ordered, non-overlapping, and gap-free; "
                f"got {previous.identifier!r} ending {previous.end} and "
                f"{current.identifier!r} starting {current.start}"
            )


def load_birth_cohort_config(path: str | Path | None) -> BirthCohortConfig:
    resolved_path = Path(path) if path is not None else DEFAULT_BIRTH_COHORT_CONFIG_PATH
    resolved_path = resolved_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Birth cohort configuration does not exist: {resolved_path}")
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Birth cohort configuration is not valid JSON: {resolved_path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Birth cohort configuration must be a JSON object")
    name, version = payload.get("name"), payload.get("version")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Birth cohort configuration requires a non-empty string 'name'")
    if not isinstance(version, int) or version < 1:
        raise ValueError("Birth cohort configuration requires an integer 'version' of at least one")
    if payload.get("time_field") != "birth_year":
        raise ValueError("This audit currently supports only time_field='birth_year'")
    missing_policy = payload.get("missing_policy")
    if missing_policy != "exclude_from_stratified_results":
        raise ValueError("missing_policy must be 'exclude_from_stratified_results'")
    raw_bins = payload.get("bins")
    if not isinstance(raw_bins, list):
        raise ValueError("Birth cohort configuration field 'bins' must be a list")
    cohorts = tuple(_parse_cohort(value) for value in raw_bins)
    if len(set(cohort.identifier for cohort in cohorts)) != len(cohorts):
        raise ValueError("Birth cohort bin IDs must be unique")
    _validate_exhaustive_ordered_bins(cohorts)
    return BirthCohortConfig(
        name=name.strip(),
        version=version,
        path=resolved_path,
        sha256=sha256_file(resolved_path),
        time_field="birth_year",
        missing_policy=missing_policy,
        cohorts=cohorts,
    )


def assign_birth_cohorts(
    birth_years: Iterable[object], config: BirthCohortConfig
) -> pd.DataFrame:
    """Assign every numeric birth year to exactly one configured cohort."""
    years = pd.to_numeric(pd.Series(list(birth_years)), errors="coerce").to_numpy(dtype=float)
    cohort_ids = np.full(len(years), MISSING_COHORT_ID, dtype=object)
    cohort_labels = np.full(len(years), MISSING_COHORT_LABEL, dtype=object)
    assigned = np.zeros(len(years), dtype=bool)
    for cohort in config.cohorts:
        mask = cohort.contains(years)
        if np.any(mask & assigned):
            raise ValueError(f"Birth cohort configuration assigns a year to multiple bins: {cohort.identifier}")
        cohort_ids[mask] = cohort.identifier
        cohort_labels[mask] = cohort.label
        assigned |= mask
    unassigned = np.isfinite(years) & ~assigned
    if unassigned.any():
        sample = sorted(set(years[unassigned].astype(int).tolist()))[:10]
        raise ValueError(f"Birth cohort configuration does not cover valid years: {sample}")
    return pd.DataFrame({
        "birth_year": years,
        "birth_cohort": cohort_ids,
        "birth_cohort_label": cohort_labels,
        "birth_year_missing": ~np.isfinite(years),
        "included_in_cohort_hypothesis": np.isfinite(years),
    })


def load_artifact_birth_cohorts(
    data_path: str | Path,
    config: BirthCohortConfig,
    expected_nodes: Optional[int] = None,
) -> pd.DataFrame:
    """Load the node table paired with an artifact and attach cohort assignments.

    Prediction exports use node indices, so exact row alignment with the graph
    artifact is part of the audit contract.
    """
    artifact_path = Path(data_path)
    nodes_path = artifact_path.parent / "nodes.csv"
    if not nodes_path.is_file():
        raise FileNotFoundError(f"The artifact's nodes.csv is required for birth cohorts: {nodes_path}")
    try:
        nodes = pd.read_csv(nodes_path, usecols=["node_id", config.time_field])
    except ValueError as error:
        raise ValueError(
            f"Node table {nodes_path} must contain node_id and {config.time_field}"
        ) from error
    if expected_nodes is not None and len(nodes) != int(expected_nodes):
        raise ValueError(
            f"Node table length {len(nodes)} differs from graph node count {expected_nodes}: {nodes_path}"
        )
    assignment = assign_birth_cohorts(nodes[config.time_field], config)
    result = nodes.reset_index(drop=True).copy()
    result.insert(0, "node_index", np.arange(len(result), dtype=np.int64))
    return pd.concat([result, assignment.drop(columns=["birth_year"])], axis=1)
