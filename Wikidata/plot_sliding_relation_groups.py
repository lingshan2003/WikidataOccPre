#!/usr/bin/env python3
"""Main Wikidata entry point for the audited annual Layer 0 share plots."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from DBpedia.plot_sliding_relation_groups import main

if __name__ == "__main__":
    main(default_output="visualization/wikidata_sliding_layer0_20th_century")
