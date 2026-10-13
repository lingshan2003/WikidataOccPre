#!/usr/bin/env python3
"""Export small main Wikidata sliding-window reports for reproducible local plots."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DBpedia.package_sliding_results import package, source_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="runs/wikidata_grouped_sliding_1900_2000_pm20_step1_alive2026_v1/grouped_pipeline_resolved_config.json")
    parser.add_argument("--output", default="artifacts/wikidata_sliding_alive2026_v1_reports.tar.gz")
    args = parser.parse_args()
    package(source_path(args.config), source_path(args.output))
