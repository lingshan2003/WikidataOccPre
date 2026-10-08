#!/usr/bin/env python3
"""Write isolated annual-window configurations; does not prepare or train graphs."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def configurations(start=1900, end=2000, half_window=20, step=1):
    if start > end or half_window < 1 or step < 1 or step > 2 * half_window:
        raise ValueError("Require start <= end, half-window >= 1, and 1 <= step <= 2 * half-window")
    if (end - start) % step:
        raise ValueError("End year must lie on the requested center-year grid")
    suffix = f"{start}_{end}_pm{half_window}_step{step}_v1"
    window_path = Path("config") / f"dbpedia_life_windows_{suffix}.json"
    pipeline_path = Path("config") / f"dbpedia_grouped_sliding_{suffix}.json"
    windows = {
        "name": f"dbpedia_life_windows_{suffix}", "version": 1,
        "description": (
            f"Centers {start} through {end}, stride {step}; inclusive windows [center-{half_window}, "
            f"center+{half_window}]. Life overlap or a sole known endpoint; not relationship event dates."
        ),
        "calendar_layout": "sliding_windows",
        "membership_rule": "life_interval_or_known_endpoint_in_period",
        "birth_field": "birth_year", "death_field": "death_year",
        "missing_date_policy": "exclude_if_both_dates_missing",
        "partial_date_policy": "include_known_endpoint_periods",
        "invalid_interval_policy": "exclude_if_death_before_birth",
        "periods": [
            {"id": f"center_{year}", "label": f"{year} ({year-half_window}-{year+half_window} CE)",
             "start": year-half_window, "end": year+half_window}
            for year in range(start, end + 1, step)
        ],
    }
    pipeline = json.loads((ROOT / "config/dbpedia_grouped_rgcn_graphmask_20y_v1.json").read_text())
    pipeline.update({
        "name": f"dbpedia_grouped_sliding_{suffix}",
        "period_config": str(window_path),
        "period_root": f"artifacts/dbpedia_life_windows_{suffix}",
        "relation_root": f"artifacts/dbpedia_grouped_sliding_{suffix}",
        "model_root": f"runs/dbpedia_grouped_sliding_{suffix}",
        "graphmask_root": f"runs_graphmask/dbpedia_grouped_sliding_{suffix}",
        "representations": ["multi_group"],
    })
    pipeline["graphmask_train"]["checkpoint_selection"] = "any-stage"
    return {window_path: windows, pipeline_path: pipeline}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1900)
    parser.add_argument("--end-year", type=int, default=2000)
    parser.add_argument("--half-window", type=int, default=20)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    configs = configurations(args.start_year, args.end_year, args.half_window, args.step)
    texts = {ROOT / p: json.dumps(c, ensure_ascii=False, indent=2) + "\n" for p, c in configs.items()}
    for path, content in texts.items():
        if path.exists() and path.read_text() != content:
            raise ValueError(f"Existing configuration differs; refusing overwrite: {path}")
    for path, content in texts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(content)
        print(path.relative_to(ROOT))
    print(f"{len(next(iter(configs.values()))['periods'])} windows; multi-group only; checkpoint selection across all training stages")


if __name__ == "__main__":
    main()
