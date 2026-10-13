#!/usr/bin/env python3
"""Freebase entry point for the shared, audited annual Layer 0 plotter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DBpedia.plot_sliding_relation_groups import main


if __name__ == "__main__":
    main(default_output="visualization/freebase_sliding_layer0_20th_century")
