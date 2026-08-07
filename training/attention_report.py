#!/usr/bin/env python3
"""Backward-compatible alias for the edge-level attention report.

Use ``run.py attention-edge-report`` for new commands.  Node-level raw
exports live in ``training.attention_node_report`` and are intentionally not
dispatched through this legacy module.
"""

from training.attention_edge_report import main


if __name__ == "__main__":
    main()
