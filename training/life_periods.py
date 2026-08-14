"""Versioned historical periods defined by overlap with a person's life span.

This is deliberately separate from :mod:`training.birth_cohorts`: a birth
cohort assigns a person to exactly one bin, whereas a person alive across a
historical boundary belongs to every overlapping life-period graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_LIFE_PERIOD_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "historical_life_periods_v2.json"
)
MISSING_DATE_POLICIES = frozenset({
    "exclude_if_birth_or_death_missing",
    "exclude_if_both_dates_missing",
})
PARTIAL_DATE_POLICIES = frozenset({
    "exclude_if_either_missing",
    "include_known_endpoint_periods",
})
INVALID_INTERVAL_POLICY = "exclude_if_death_before_birth"
MEMBERSHIP_RULES = frozenset({
    "life_interval_overlaps_period",
    "life_interval_or_known_endpoint_in_period",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LifePeriod:
    identifier: str
    label: str
    start: Optional[int]
    end: Optional[int]

    def manifest(self) -> Dict[str, object]:
        result: Dict[str, object] = {"id": self.identifier, "label": self.label}
        if self.start is not None:
            result["start"] = self.start
        if self.end is not None:
            result["end"] = self.end
        return result


@dataclass(frozen=True)
class LifePeriodConfig:
    name: str
    version: int
    path: Path
    sha256: str
    birth_field: str
    death_field: str
    membership_rule: str
    missing_date_policy: str
    partial_date_policy: str
    invalid_interval_policy: str
    periods: Tuple[LifePeriod, ...]

    @property
    def identifiers(self) -> Tuple[str, ...]:
        return tuple(period.identifier for period in self.periods)

    def period(self, identifier: str) -> LifePeriod:
        for period in self.periods:
            if period.identifier == identifier:
                return period
        raise ValueError(f"Unknown life period {identifier!r}. Available: {list(self.identifiers)}")

    def manifest(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "version": self.version,
            "path": str(self.path),
            "sha256": self.sha256,
            "membership_rule": self.membership_rule,
            "birth_field": self.birth_field,
            "death_field": self.death_field,
            "missing_date_policy": self.missing_date_policy,
            "partial_date_policy": self.partial_date_policy,
            "invalid_interval_policy": self.invalid_interval_policy,
            "periods": [period.manifest() for period in self.periods],
        }


def _optional_integer(value: object, field: str, period_id: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Life period {period_id!r} field {field!r} must be an integer when provided")
    return value


def _parse_period(value: object) -> LifePeriod:
    if not isinstance(value, Mapping):
        raise ValueError("Every life-period entry must be a JSON object")
    identifier, label = value.get("id"), value.get("label")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("Every life-period entry requires a non-empty string 'id'")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"Life period {identifier!r} requires a non-empty string 'label'")
    start = _optional_integer(value.get("start"), "start", identifier)
    end = _optional_integer(value.get("end"), "end", identifier)
    if start is not None and end is not None and start > end:
        raise ValueError(f"Life period {identifier!r} has start greater than end")
    return LifePeriod(identifier.strip(), label.strip(), start, end)


def _validate_partition(periods: Sequence[LifePeriod]) -> None:
    """Validate a contiguous historical timeline, while memberships may overlap."""
    if not periods:
        raise ValueError("Life-period configuration requires at least one period")
    if periods[0].start is not None or periods[-1].end is not None:
        raise ValueError("Life periods must cover all years: first start and final end must be open")
    for previous, current in zip(periods, periods[1:]):
        if previous.end is None or current.start is None or current.start != previous.end + 1:
            raise ValueError("Life periods must be ordered, non-overlapping, and gap-free in calendar time")


def load_life_period_config(path: str | Path | None) -> LifePeriodConfig:
    resolved = Path(path) if path is not None else DEFAULT_LIFE_PERIOD_CONFIG_PATH
    resolved = resolved.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Life-period configuration does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Life-period configuration is not valid JSON: {resolved}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("Life-period configuration must be a JSON object")
    if not isinstance(payload.get("name"), str) or not payload["name"].strip():
        raise ValueError("Life-period configuration requires a non-empty string 'name'")
    if not isinstance(payload.get("version"), int) or payload["version"] < 1:
        raise ValueError("Life-period configuration requires integer version >= 1")
    membership_rule = payload.get("membership_rule")
    if membership_rule not in MEMBERSHIP_RULES:
        raise ValueError(f"membership_rule must be one of {sorted(MEMBERSHIP_RULES)}")
    if payload.get("birth_field") != "birth_year" or payload.get("death_field") != "death_year":
        raise ValueError("This experiment requires birth_field='birth_year' and death_field='death_year'")
    missing_date_policy = payload.get("missing_date_policy")
    if missing_date_policy not in MISSING_DATE_POLICIES:
        raise ValueError(f"missing_date_policy must be one of {sorted(MISSING_DATE_POLICIES)}")
    partial_date_policy = payload.get("partial_date_policy")
    if partial_date_policy is None and missing_date_policy == "exclude_if_birth_or_death_missing":
        # Keep v1 artifacts readable, but do not make this the default for
        # new experiments.
        partial_date_policy = "exclude_if_either_missing"
    if partial_date_policy not in PARTIAL_DATE_POLICIES:
        raise ValueError(f"partial_date_policy must be one of {sorted(PARTIAL_DATE_POLICIES)}")
    if membership_rule == "life_interval_overlaps_period" and partial_date_policy != "exclude_if_either_missing":
        raise ValueError("life_interval_overlaps_period requires partial_date_policy='exclude_if_either_missing'")
    if membership_rule == "life_interval_or_known_endpoint_in_period" and partial_date_policy != "include_known_endpoint_periods":
        raise ValueError(
            "life_interval_or_known_endpoint_in_period requires "
            "partial_date_policy='include_known_endpoint_periods'"
        )
    if payload.get("invalid_interval_policy") != INVALID_INTERVAL_POLICY:
        raise ValueError(f"invalid_interval_policy must be {INVALID_INTERVAL_POLICY!r}")
    raw_periods = payload.get("periods")
    if not isinstance(raw_periods, list):
        raise ValueError("Life-period configuration field 'periods' must be a list")
    periods = tuple(_parse_period(value) for value in raw_periods)
    if len(set(period.identifier for period in periods)) != len(periods):
        raise ValueError("Life-period IDs must be unique")
    _validate_partition(periods)
    return LifePeriodConfig(
        name=payload["name"].strip(),
        version=payload["version"],
        path=resolved,
        sha256=sha256_file(resolved),
        birth_field="birth_year",
        death_field="death_year",
        membership_rule=str(membership_rule),
        missing_date_policy=str(missing_date_policy),
        partial_date_policy=str(partial_date_policy),
        invalid_interval_policy=INVALID_INTERVAL_POLICY,
        periods=periods,
    )


def life_period_membership(nodes: pd.DataFrame, config: LifePeriodConfig) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Return one Boolean node mask per period and a transparent date audit.

    A valid life interval ``[birth, death]`` belongs to period ``[start, end]``
    exactly when the intervals intersect. For the v2 policy, a person with
    only a known birth (or only a known death) is assigned to the single period
    containing that observed endpoint; no unobserved years are extrapolated.
    """
    required = {config.birth_field, config.death_field}
    missing = required - set(nodes)
    if missing:
        raise ValueError(f"Node table lacks required life-date columns: {sorted(missing)}")
    birth = pd.to_numeric(nodes[config.birth_field], errors="coerce").to_numpy(dtype=float)
    death = pd.to_numeric(nodes[config.death_field], errors="coerce").to_numpy(dtype=float)
    has_birth, has_death = np.isfinite(birth), np.isfinite(death)
    valid_life_interval = has_birth & has_death & (death >= birth)
    birth_endpoint_only = has_birth & ~has_death
    death_endpoint_only = ~has_birth & has_death
    status = np.full(len(nodes), "valid_life_interval", dtype=object)
    status[~has_birth & ~has_death] = "missing_birth_and_death"
    status[death_endpoint_only] = "death_endpoint_only"
    status[birth_endpoint_only] = "birth_endpoint_only"
    status[has_birth & has_death & (death < birth)] = "death_before_birth"

    memberships: Dict[str, np.ndarray] = {}
    for period in config.periods:
        mask = valid_life_interval.copy()
        if period.end is not None:
            mask &= birth <= period.end
        if period.start is not None:
            mask &= death >= period.start
        if config.partial_date_policy == "include_known_endpoint_periods":
            birth_in_period = birth_endpoint_only.copy()
            death_in_period = death_endpoint_only.copy()
            if period.start is not None:
                birth_in_period &= birth >= period.start
                death_in_period &= death >= period.start
            if period.end is not None:
                birth_in_period &= birth <= period.end
                death_in_period &= death <= period.end
            mask |= birth_in_period | death_in_period
        memberships[period.identifier] = mask
    eligible = valid_life_interval | birth_endpoint_only | death_endpoint_only
    if config.partial_date_policy == "exclude_if_either_missing":
        eligible = valid_life_interval
    audit = pd.DataFrame({
        "birth_year_numeric": birth,
        "death_year_numeric": death,
        "life_interval_status": status,
        "eligible_for_life_period_experiment": eligible,
        "life_period_membership_count": np.sum(np.stack(list(memberships.values())), axis=0),
    })
    return memberships, audit
