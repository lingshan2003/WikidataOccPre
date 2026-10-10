#!/usr/bin/env python3
"""Write isolated Freebase annual-window configurations; do not prepare/train."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = ROOT / "config/freebase_grouped_rgcn_graphmask_20y_v2.json"


def configurations(start=1900, end=2000, half_window=20, step=1, *,
                   assume_alive_through=None, born_on_or_after=1920):
    if start > end or half_window < 1 or step < 1 or step > 2 * half_window:
        raise ValueError("Require start <= end, half-window >= 1, and 1 <= step <= 2 * half-window")
    if (end - start) % step:
        raise ValueError("End year must lie on the requested center-year grid")

    if assume_alive_through is not None:
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (assume_alive_through, born_on_or_after)):
            raise ValueError("Alive-through and inclusive birth threshold must be integer years")
        if assume_alive_through < born_on_or_after:
            raise ValueError("Alive-through year must not precede the inclusive birth threshold")
    suffix = f"{start}_{end}_pm{half_window}_step{step}"
    suffix += ("_v2" if assume_alive_through is None else
               f"_born_ge{born_on_or_after}_alive{assume_alive_through}_v3")
    period_path = Path("config") / f"freebase_life_windows_{suffix}.json"
    pipeline_path = Path("config") / f"freebase_grouped_sliding_{suffix}.json"
    periods = {
        "name": f"freebase_life_windows_{suffix}",
        "version": 1,
        "description": (
            f"Centers {start} through {end}, stride {step}; inclusive windows [center-{half_window}, "
            f"center+{half_window}]. Include nodes whose observed valid life interval intersects a window, "
            "or whose sole known birth/death year falls in it. These are person-life windows, not "
            "relationship event dates."
        ),
        "calendar_layout": "sliding_windows",
        "membership_rule": "life_interval_or_known_endpoint_in_period",
        "birth_field": "birth_year",
        "death_field": "death_year",
        "missing_date_policy": "exclude_if_both_dates_missing",
        "partial_date_policy": "include_known_endpoint_periods",
        "invalid_interval_policy": "exclude_if_death_before_birth",
        "periods": [
            {
                "id": f"center_{year}",
                "label": f"{year} ({year-half_window}-{year+half_window} CE)",
                "start": year - half_window,
                "end": year + half_window,
            }
            for year in range(start, end + 1, step)
        ],
    }
    if assume_alive_through is not None:
        periods["version"] = 2
        # The shared schema uses a strict comparison; translate the inclusive
        # Freebase rule rather than changing DBpedia's existing semantics.
        periods["birth_only_alive_assumption"] = {
            "born_after": born_on_or_after - 1, "alive_through": assume_alive_through,
        }
        periods["description"] += (
            f" For birth >= {born_on_or_after} with no observed death year, assume a life interval "
            f"ending in {assume_alive_through} for membership only. Raw death years and temporal "
            "feature inputs remain unchanged; observed deaths are never extended."
        )

    pipeline = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    pipeline.update({
        "name": f"freebase_grouped_sliding_{suffix}",
        "source_data": "artifacts/freebase_provisional_l1_v2/graph_data.pt",
        "period_config": str(period_path),
        "period_root": f"artifacts/freebase_life_windows_{suffix}",
        "relation_root": f"artifacts/freebase_grouped_sliding_{suffix}",
        "model_root": f"runs/freebase_grouped_sliding_{suffix}",
        "graphmask_root": f"runs_graphmask/freebase_grouped_sliding_{suffix}",
        "periods": "all",
        "representations": ["multi_group"],
        "device": None,
    })
    return {period_path: periods, pipeline_path: pipeline}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1900)
    parser.add_argument("--end-year", type=int, default=2000)
    parser.add_argument("--half-window", type=int, default=20)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--assume-alive-through", type=int, help="Opt in to a separately versioned life-extension assumption")
    parser.add_argument("--born-on-or-after", type=int, default=1920, help="Inclusive birth threshold for the optional assumption")
    args = parser.parse_args()
    configs = configurations(args.start_year, args.end_year, args.half_window, args.step,
                             assume_alive_through=args.assume_alive_through, born_on_or_after=args.born_on_or_after)
    texts = {
        ROOT / path: json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        for path, payload in configs.items()
    }
    for path, content in texts.items():
        if path.exists() and path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing configuration differs; refusing overwrite: {path}")
    for path, content in texts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        print(path.relative_to(ROOT))
    print(f"{len(next(iter(configs.values()))['periods'])} windows; multi_group only; device remains unset")


if __name__ == "__main__":
    main()
