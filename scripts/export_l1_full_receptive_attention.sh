#!/usr/bin/env bash
# Re-evaluate completed L1 RGAT checkpoints with every root's complete
# message-passing receptive field, while preserving per-root occupation masks.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
DATA_PATH="${RGCN_L1_DATA:-artifacts/level1_hierarchy/graph_data.pt}"
OUTPUT_ROOT="${RGCN_L1_REPORT_ROOT:-runs_report/level1}"
OUTPUT_DIR="${RGCN_L1_FULL_ATTENTION_OUTPUT:-$OUTPUT_ROOT/rgat_l1_full_receptive_attention_all_relations}"
BATCH_SIZE="${RGCN_L1_FULL_ATTENTION_BATCH_SIZE:-32}"
MATRIX_RELATIONS="${RGCN_L1_FULL_ATTENTION_RELATIONS:-all}"
SEEDS=(42 43 44)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/export_l1_full_receptive_attention.sh plan
  bash scripts/export_l1_full_receptive_attention.sh run

This does not train. It treats every retained-L1 person as a prediction root,
masks that root's own occupation in its batch, and uses all neighbours at each
message-passing layer (-1 for one-layer checkpoints; -1,-1 for two-layer).

Environment overrides:
  RGCN_L1_DATA=artifacts/level1_hierarchy/graph_data.pt
  RGCN_L1_REPORT_ROOT=runs_report/level1
  RGCN_L1_FULL_ATTENTION_OUTPUT=runs_report/level1/rgat_l1_full_receptive_attention_all_relations
  RGCN_L1_FULL_ATTENTION_BATCH_SIZE=32
  RGCN_L1_FULL_ATTENTION_RELATIONS=all  # all exact directed relations (default)
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then
  usage
  exit 2
fi

show_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

command=(
  python run.py attention-edge-report
  --data "$DATA_PATH"
  --checkpoint-glob "$OUTPUT_ROOT/rgat_one_hop/seed_*/best_model.pt"
  --checkpoint-glob "$OUTPUT_ROOT/rgat_baseline/seed_*/best_model.pt"
  --output-dir "$OUTPUT_DIR"
  --split labeled
  --num-neighbors full
  --batch-size "$BATCH_SIZE"
  --num-workers 0
  --occupation-matrix-relations "$MATRIX_RELATIONS"
  --matrix-min-edge-count 10
  --device cuda
)

if [[ "$MODE" == "plan" ]]; then
  printf '%-28s' 'full_receptive_attention'
  show_command "${command[@]}"
  exit 0
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Missing L1 artifact: $DATA_PATH" >&2
  exit 1
fi
for experiment in rgat_one_hop rgat_baseline; do
  for seed in "${SEEDS[@]}"; do
    checkpoint="$OUTPUT_ROOT/$experiment/seed_$seed/best_model.pt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing checkpoint: $checkpoint" >&2
      exit 1
    fi
  done
done

"${command[@]}"
