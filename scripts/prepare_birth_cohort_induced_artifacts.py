#!/usr/bin/env python3
"""Deprecated wrapper: use prepare_life_period_induced_artifacts.py instead."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.birth_cohort_artifacts import main


if __name__ == "__main__":
    main()
