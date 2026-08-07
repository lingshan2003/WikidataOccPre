#!/usr/bin/env bash
# Export test-root direct alpha, two-hop rollout and root-bootstrap summaries.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
DATA_PATH="${RGCN_L1_DATA:-artifacts/level1_hierarchy/graph_data.pt}"
OUTPUT_ROOT="${RGCN_L1_REPORT_ROOT:-runs_report/level1}"
OUTPUT_DIR="${RGCN_L1_ROOT_ATTENTION_OUTPUT:-$OUTPUT_ROOT/rgat_l1_root_attention_all_relations}"
FORWARD_MODE="${RGCN_L1_ATTENTION_FORWARD_MODE:-full-graph}"
BATCH_SIZE="${RGCN_L1_ATTENTION_BATCH_SIZE:-32}"
BOOTSTRAP_RESAMPLES="${RGCN_L1_ATTENTION_BOOTSTRAP_RESAMPLES:-2000}"
SEEDS=(42 43 44)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/export_l1_root_attention.sh plan
  bash scripts/export_l1_root_attention.sh run

Exports all exact directed direct relations and all ordered two-hop relation
pairs for test roots only. The default is one full-graph eval forward per
checkpoint. Set RGCN_L1_ATTENTION_FORWARD_MODE=full-neighborhood to replay
complete (-1) receptive-field batches when full graph inference does not fit.

Environment overrides:
  RGCN_L1_DATA=artifacts/level1_hierarchy/graph_data.pt
  RGCN_L1_REPORT_ROOT=runs_report/level1
  RGCN_L1_ROOT_ATTENTION_OUTPUT=runs_report/level1/rgat_l1_root_attention_all_relations
  RGCN_L1_ATTENTION_FORWARD_MODE=full-graph|full-neighborhood
  RGCN_L1_ATTENTION_BATCH_SIZE=32
  RGCN_L1_ATTENTION_BOOTSTRAP_RESAMPLES=2000
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then
  usage
  exit 2
fi
if [[ "$FORWARD_MODE" != "full-graph" && "$FORWARD_MODE" != "full-neighborhood" ]]; then
  echo "RGCN_L1_ATTENTION_FORWARD_MODE must be full-graph or full-neighborhood" >&2
  exit 2
fi

show_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

direct_command=(
  python run.py attention-node-report
  --data "$DATA_PATH"
  --checkpoint-glob "$OUTPUT_ROOT/rgat_one_hop/seed_*/best_model.pt"
  --checkpoint-glob "$OUTPUT_ROOT/rgat_baseline/seed_*/best_model.pt"
  --checkpoint-glob "$OUTPUT_ROOT/rgat_l1_only_one_hop/seed_*/best_model.pt"
  --checkpoint-glob "$OUTPUT_ROOT/rgat_l1_only_two_hop/seed_*/best_model.pt"
  --output-dir "$OUTPUT_DIR/direct"
  --split test
  --forward-mode "$FORWARD_MODE"
  --num-neighbors full
  --batch-size "$BATCH_SIZE"
  --num-workers 0
  --device cuda
)

direct_bootstrap_command=(
  python run.py attention-bootstrap
  --roster "$OUTPUT_DIR/direct/root_attention_roster_by_seed.csv.gz"
  --sparse "$OUTPUT_DIR/direct/root_direct_attention_sparse_by_seed.csv.gz"
  --output "$OUTPUT_DIR/direct/root_direct_attention_bootstrap.csv"
  --resamples "$BOOTSTRAP_RESAMPLES"
)

rollout_command=(
  python run.py attention-rollout-report
  --data "$DATA_PATH"
  --checkpoint-glob "$OUTPUT_ROOT/rgat_baseline/seed_*/best_model.pt"
  --checkpoint-glob "$OUTPUT_ROOT/rgat_l1_only_two_hop/seed_*/best_model.pt"
  --output-dir "$OUTPUT_DIR/rollout"
  --split test
  --forward-mode "$FORWARD_MODE"
  --num-neighbors full
  --batch-size "$BATCH_SIZE"
  --num-workers 0
  --device cuda
)

rollout_bootstrap_command=(
  python run.py attention-bootstrap
  --roster "$OUTPUT_DIR/rollout/root_two_hop_rollout_roster_by_seed.csv.gz"
  --sparse "$OUTPUT_DIR/rollout/root_two_hop_rollout_sparse_by_seed.csv.gz"
  --output "$OUTPUT_DIR/rollout/root_two_hop_rollout_bootstrap.csv"
  --resamples "$BOOTSTRAP_RESAMPLES"
)

if [[ "$MODE" == "plan" ]]; then
  printf '%-28s' 'direct_attention'
  show_command "${direct_command[@]}"
  printf '%-28s' 'direct_bootstrap'
  show_command "${direct_bootstrap_command[@]}"
  printf '%-28s' 'two_hop_rollout'
  show_command "${rollout_command[@]}"
  printf '%-28s' 'rollout_bootstrap'
  show_command "${rollout_bootstrap_command[@]}"
  exit 0
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Missing L1 artifact: $DATA_PATH" >&2
  exit 1
fi
for experiment in rgat_one_hop rgat_baseline rgat_l1_only_one_hop rgat_l1_only_two_hop; do
  for seed in "${SEEDS[@]}"; do
    checkpoint="$OUTPUT_ROOT/$experiment/seed_$seed/best_model.pt"
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing checkpoint: $checkpoint" >&2
      exit 1
    fi
  done
done

"${direct_command[@]}"
"${direct_bootstrap_command[@]}"
"${rollout_command[@]}"
"${rollout_bootstrap_command[@]}"
