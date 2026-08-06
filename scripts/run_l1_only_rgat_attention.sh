#!/usr/bin/env bash
# Train the L1-only one-hop and two-hop RGAT controls used by the attention study.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
DATA_PATH="${RGCN_L1_DATA:-artifacts/level1_hierarchy/graph_data.pt}"
OUTPUT_ROOT="${RGCN_L1_REPORT_ROOT:-runs_report/level1}"
SEEDS=(42 43 44)

COMMON_ARGS=(
  --model rgat
  --data "$DATA_PATH"
  --epochs 50
  --batch-size 512
  --hidden-dim 128
  --branch-dim 64
  --heads 4
  --early-stop-metric macro_f1
  --min-delta 0.001
  --patience 6
  --num-workers 4
  --device cuda
  --occupation-feature-levels 1
  --auxiliary-features none
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_l1_only_rgat_attention.sh plan
  bash scripts/run_l1_only_rgat_attention.sh run

Environment overrides:
  RGCN_L1_DATA=artifacts/level1_hierarchy/graph_data.pt
  RGCN_L1_REPORT_ROOT=runs_report/level1

The script trains three seeds for each L1-only condition. The only intended
difference between conditions is message-passing depth and its matching fanout.
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

run_condition() {
  local experiment="$1"
  local layers="$2"
  local fanouts="$3"
  for seed in "${SEEDS[@]}"; do
    local output_dir="$OUTPUT_ROOT/$experiment/seed_$seed"
    local command=(
      python run.py train "${COMMON_ARGS[@]}"
      --output-dir "$output_dir"
      --num-layers "$layers"
      --num-neighbors "$fanouts"
      --seed "$seed"
    )
    if [[ "$MODE" == "plan" ]]; then
      printf '%-26s seed=%s' "$experiment" "$seed"
      show_command "${command[@]}"
      continue
    fi
    if [[ ! -f "$DATA_PATH" ]]; then
      echo "Missing L1 artifact: $DATA_PATH" >&2
      exit 1
    fi
    if [[ -f "$output_dir/metrics.json" ]]; then
      echo "[skip] $experiment seed=$seed already has $output_dir/metrics.json"
      continue
    fi
    mkdir -p "$output_dir"
    printf '%q ' "${command[@]}" > "$output_dir/command.sh"
    printf '\n' >> "$output_dir/command.sh"
    echo "[run] $experiment seed=$seed"
    "${command[@]}"
  done
}

run_condition rgat_l1_only_one_hop 1 15
run_condition rgat_l1_only_two_hop 2 15,10

if [[ "$MODE" == "plan" ]]; then
  echo "Printed 2 L1-only RGAT conditions × 3 seeds = 6 training commands."
fi
