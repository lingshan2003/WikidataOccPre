#!/usr/bin/env bash
# Train the L1 one-hop RGAT control and export its relation-attention table
# alongside the existing two-hop RGAT report baseline.

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
  --num-neighbors 15
  --num-layers 1
  --hidden-dim 128
  --branch-dim 64
  --heads 4
  --early-stop-metric macro_f1
  --min-delta 0.001
  --patience 6
  --num-workers 4
  --device cuda
  --occupation-feature-levels 1,2,3
  --auxiliary-features none
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_l1_rgat_attention.sh plan
  bash scripts/run_l1_rgat_attention.sh run

Environment overrides:
  RGCN_L1_DATA=artifacts/level1_hierarchy/graph_data.pt
  RGCN_L1_REPORT_ROOT=runs_report/level1

The existing two-hop checkpoints must be at:
  $RGCN_L1_REPORT_ROOT/rgat_baseline/seed_{42,43,44}/best_model.pt
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

for seed in "${SEEDS[@]}"; do
  output_dir="$OUTPUT_ROOT/rgat_one_hop/seed_$seed"
  command=(python run.py train "${COMMON_ARGS[@]}" --output-dir "$output_dir" --seed "$seed")
  if [[ "$MODE" == "plan" ]]; then
    printf '%-18s seed=%s' 'rgat_one_hop' "$seed"
    show_command "${command[@]}"
  elif [[ -f "$output_dir/metrics.json" ]]; then
    echo "[skip] rgat_one_hop seed=$seed already has $output_dir/metrics.json"
  else
    if [[ ! -f "$DATA_PATH" ]]; then
      echo "Missing L1 artifact: $DATA_PATH" >&2
      exit 1
    fi
    mkdir -p "$output_dir"
    printf '%q ' "${command[@]}" > "$output_dir/command.sh"
    printf '\n' >> "$output_dir/command.sh"
    echo "[run] rgat_one_hop seed=$seed"
    "${command[@]}"
  fi
done

attention_command=(
  python run.py attention-report
  --data "$DATA_PATH"
  --checkpoint-glob "$OUTPUT_ROOT/rgat_one_hop/seed_*/best_model.pt"
  --checkpoint-glob "$OUTPUT_ROOT/rgat_baseline/seed_*/best_model.pt"
  --output-dir "$OUTPUT_ROOT/rgat_l1_relation_attention"
  --split test
  --num-neighbors auto
  --batch-size 512
  --num-workers 0
  --device cuda
)

if [[ "$MODE" == "plan" ]]; then
  printf '%-18s' 'attention_report'
  show_command "${attention_command[@]}"
  echo "Run mode trains three one-hop seeds, then writes relation_attention_table.md/.csv."
else
  for seed in "${SEEDS[@]}"; do
    baseline="$OUTPUT_ROOT/rgat_baseline/seed_$seed/best_model.pt"
    if [[ ! -f "$baseline" ]]; then
      echo "Missing existing two-hop baseline checkpoint: $baseline" >&2
      exit 1
    fi
  done
  "${attention_command[@]}"
  python scripts/summarize_report_runs.py --root "$OUTPUT_ROOT"
fi
