#!/usr/bin/env bash
# Export direct root-level RGAT gradient × attention for the one-hop model.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
DATA_PATH="${RGCN_L1_DATA:-artifacts/level1_hierarchy/graph_data.pt}"
REPORT_ROOT="${RGCN_L1_REPORT_ROOT:-runs_report/level1}"
OUTPUT_DIR="${RGCN_L1_GRADIENT_OUTPUT:-$REPORT_ROOT/rgat_l1_root_attention_all_relations/gradient_x_attention/rgat_one_hop}"
FORWARD_MODE="${RGCN_L1_GRADIENT_FORWARD_MODE:-full-neighborhood}"
BATCH_SIZE="${RGCN_L1_GRADIENT_BATCH_SIZE:-16}"
SCORE="${RGCN_L1_GRADIENT_SCORE:-predicted-margin}"
BOOTSTRAP_RESAMPLES="${RGCN_L1_GRADIENT_BOOTSTRAP_RESAMPLES:-2000}"
SEEDS=(42 43 44)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/export_rgat_one_hop_gradient_x_attention.sh plan
  bash scripts/export_rgat_one_hop_gradient_x_attention.sh run

Exports root-level, signed alpha * d(prediction score)/d(alpha) for the three
rgat_one_hop checkpoints. Each sparse row is one root × exact directed
relation × source-L1 × visibility group and also contains the group's summed
attention mass. The paired bootstrap reconstructs absent root/groups as zero.

Environment overrides:
  RGCN_L1_DATA=artifacts/level1_hierarchy/graph_data.pt
  RGCN_L1_REPORT_ROOT=runs_report/level1
  RGCN_L1_GRADIENT_OUTPUT=...
  RGCN_L1_GRADIENT_FORWARD_MODE=full-neighborhood|full-graph
  RGCN_L1_GRADIENT_BATCH_SIZE=16
  RGCN_L1_GRADIENT_SCORE=predicted-margin|true-margin|predicted-logit|true-logit
  RGCN_L1_GRADIENT_BOOTSTRAP_RESAMPLES=2000
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then
  usage
  exit 2
fi
if [[ "$FORWARD_MODE" != "full-graph" && "$FORWARD_MODE" != "full-neighborhood" ]]; then
  echo "RGCN_L1_GRADIENT_FORWARD_MODE must be full-graph or full-neighborhood" >&2
  exit 2
fi

show_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

gradient_command=(
  python run.py gradient-attribution-report
  --data "$DATA_PATH"
  --checkpoint-glob "$REPORT_ROOT/rgat_one_hop/seed_*/best_model.pt"
  --output-dir "$OUTPUT_DIR"
  --split test
  --score "$SCORE"
  --forward-mode "$FORWARD_MODE"
  --num-neighbors full
  --batch-size "$BATCH_SIZE"
  --num-workers 0
  --device cuda
)

bootstrap_command=(
  python run.py attention-bootstrap
  --roster "$OUTPUT_DIR/root_gradient_x_attention_roster_by_seed.csv.gz"
  --sparse "$OUTPUT_DIR/root_gradient_x_attention_sparse_by_seed.csv.gz"
  --output "$OUTPUT_DIR/root_gradient_x_attention_bootstrap.csv"
  --value-columns gradient_x_attention,absolute_gradient_x_attention_sum,attention_mass
  --group-columns relation_id,relation,source_l1_id,source_l1,source_visibility
  --resamples "$BOOTSTRAP_RESAMPLES"
)

if [[ "$MODE" == "plan" ]]; then
  printf '%-28s' 'gradient_x_attention'
  show_command "${gradient_command[@]}"
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

"${gradient_command[@]}"
"${bootstrap_command[@]}"
