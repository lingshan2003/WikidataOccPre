#!/usr/bin/env bash
# Sweep one-hop conditional occupation-pair counterfactual ablations.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
DATA_PATH="${RGCN_L1_DATA:-artifacts/level1_hierarchy/graph_data.pt}"
REPORT_ROOT="${RGCN_L1_REPORT_ROOT:-runs_report/level1}"
OUTPUT_DIR="${RGCN_PAIR_ABLATION_OUTPUT:-$REPORT_ROOT/relation_pair_sweep}"
CONTROL_DRAWS="${RGCN_PAIR_CONTROL_DRAWS:-10}"
MAX_ROOTS="${RGCN_PAIR_MAX_ROOTS:-}"
MIN_SUMMARY_ROOTS="${RGCN_PAIR_MIN_SUMMARY_ROOTS:-10}"
DEVICE="${RGCN_PAIR_DEVICE:-cuda}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_l1_relation_pair_ablation.sh plan
  bash scripts/run_l1_relation_pair_ablation.sh run

The command holds the one-hop RGAT checkpoints fixed and sweeps every observed
(visible source L1, exact directed relation, true target L1) motif. For each
root-level motif, its matched control removes the same number of other direct
incoming messages from visible sources with the same L1 at that same root.

Environment overrides:
  RGCN_L1_DATA=artifacts/level1_hierarchy/graph_data.pt
  RGCN_L1_REPORT_ROOT=runs_report/level1
  RGCN_PAIR_CONTROL_DRAWS=10
  RGCN_PAIR_MAX_ROOTS=2000    # optional quick pilot; omit for all roots
  RGCN_PAIR_MIN_SUMMARY_ROOTS=10
  RGCN_PAIR_DEVICE=cuda

The checkpoint must be one-layer RGAT and available at:
  $RGCN_L1_REPORT_ROOT/rgat_one_hop/seed_{42,43,44}/best_model.pt
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then
  usage
  exit 2
fi

command=(
  .venv/bin/python run.py relation-pair-sweep-report
  --data "$DATA_PATH"
  --checkpoint-glob "$REPORT_ROOT/rgat_one_hop/seed_*/best_model.pt"
  --output-dir "$OUTPUT_DIR"
  --split test
  --forward-mode full-neighborhood
  --num-neighbors full
  --batch-size 64
  --num-workers 0
  --control-draws "$CONTROL_DRAWS"
  --min-summary-roots "$MIN_SUMMARY_ROOTS"
  --device "$DEVICE"
)
if [[ -n "$MAX_ROOTS" ]]; then
  command+=(--max-roots "$MAX_ROOTS")
fi

if [[ "$MODE" == "plan" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  echo "Run mode writes every observed relation pair's root-level paired margins and per-seed summaries; it does not retrain RGAT."
  exit 0
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Missing L1 artifact: $DATA_PATH" >&2
  exit 1
fi
for seed in 42 43 44; do
  checkpoint="$REPORT_ROOT/rgat_one_hop/seed_$seed/best_model.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing one-hop checkpoint: $checkpoint" >&2
    exit 1
  fi
done

"${command[@]}"
