#!/usr/bin/env bash
# Export value-aware alpha * W_r * h_j message magnitudes for RGAT one-hop.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
DATA_PATH="${RGCN_L1_DATA:-artifacts/level1_hierarchy/graph_data.pt}"
REPORT_ROOT="${RGCN_L1_REPORT_ROOT:-runs_report/level1}"
OUTPUT_DIR="${RGCN_L1_MESSAGE_OUTPUT:-$REPORT_ROOT/rgat_l1_root_attention_all_relations/message_contribution/rgat_one_hop}"
FORWARD_MODE="${RGCN_L1_MESSAGE_FORWARD_MODE:-full-neighborhood}"
BATCH_SIZE="${RGCN_L1_MESSAGE_BATCH_SIZE:-16}"
BOOTSTRAP_RESAMPLES="${RGCN_L1_MESSAGE_BOOTSTRAP_RESAMPLES:-2000}"
SEEDS=(42 43 44)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/export_rgat_one_hop_message_contribution.sh plan
  bash scripts/export_rgat_one_hop_message_contribution.sh run

Exports root-level value-aware message contributions for the three rgat_one_hop
checkpoints.  For each typed edge, the vector before RGAT aggregation is
mean_head(alpha * (h_source @ W_relation)).  The report exports both the L2
norm of the summed group vector and the sum of edgewise L2 norms.

Environment overrides:
  RGCN_L1_DATA=artifacts/level1_hierarchy/graph_data.pt
  RGCN_L1_REPORT_ROOT=runs_report/level1
  RGCN_L1_MESSAGE_OUTPUT=...
  RGCN_L1_MESSAGE_FORWARD_MODE=full-neighborhood|full-graph
  RGCN_L1_MESSAGE_BATCH_SIZE=16
  RGCN_L1_MESSAGE_BOOTSTRAP_RESAMPLES=2000
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then
  usage
  exit 2
fi
if [[ "$FORWARD_MODE" != "full-graph" && "$FORWARD_MODE" != "full-neighborhood" ]]; then
  echo "RGCN_L1_MESSAGE_FORWARD_MODE must be full-graph or full-neighborhood" >&2
  exit 2
fi

show_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

message_command=(
  python run.py message-contribution-report
  --data "$DATA_PATH"
  --checkpoint-glob "$REPORT_ROOT/rgat_one_hop/seed_*/best_model.pt"
  --output-dir "$OUTPUT_DIR"
  --split test
  --forward-mode "$FORWARD_MODE"
  --num-neighbors full
  --batch-size "$BATCH_SIZE"
  --num-workers 0
  --device cuda
)

bootstrap_command=(
  python run.py attention-bootstrap
  --roster "$OUTPUT_DIR/root_message_contribution_roster_by_seed.csv.gz"
  --sparse "$OUTPUT_DIR/root_message_contribution_sparse_by_seed.csv.gz"
  --output "$OUTPUT_DIR/root_message_contribution_bootstrap.csv"
  --value-columns message_contribution_l2,absolute_message_l2_sum,message_contribution_l2_share
  --group-columns relation_id,relation,source_l1_id,source_l1,source_visibility
  --resamples "$BOOTSTRAP_RESAMPLES"
)

if [[ "$MODE" == "plan" ]]; then
  printf '%-28s' 'message_contribution'
  show_command "${message_command[@]}"
  printf '%-28s' 'root_bootstrap'
  show_command "${bootstrap_command[@]}"
  exit 0
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Missing L1 artifact: $DATA_PATH" >&2
  exit 1
fi
for seed in "${SEEDS[@]}"; do
  checkpoint="$REPORT_ROOT/rgat_one_hop/seed_$seed/best_model.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint: $checkpoint" >&2
    exit 1
  fi
done

"${message_command[@]}"
"${bootstrap_command[@]}"
