#!/usr/bin/env python3
"""Package completed Freebase annual reports with the shared lightweight exporter."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DBpedia.package_sliding_results import package, source_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/freebase_grouped_sliding_1900_2000_pm20_step1_born_ge1920_alive2026_v3.json")
    parser.add_argument("--output", default="artifacts/freebase_sliding_alive2026_v3_reports.tar.gz")
    args = parser.parse_args()
    package(source_path(args.config), source_path(args.output))


if __name__ == "__main__":
    main()
