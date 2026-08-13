#!/usr/bin/env bash
# Tomorrow's Level-1 inherited/acquired audit.
#
# Reuses the completed rgcn_baseline and rgat_baseline runs under
# BASELINE_ROOT. It never schedules those six full-graph runs again. The only
# training work is the four ablation/control conditions for each model and
# seed: 2 models x 4 conditions x 3 seeds = 24 runs.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
GROUP="${2:-all}"
PYTHON_BIN="${RGCN_PYTHON_BIN:-python}"
DATA_PATH="${RGCN_TIE_AUDIT_DATA:-artifacts/level1_hierarchy/graph_data.pt}"
BASELINE_ROOT="${RGCN_TIE_AUDIT_BASELINE_ROOT:-runs_report/level1}"
OUTPUT_ROOT="${RGCN_TIE_AUDIT_OUTPUT_ROOT:-runs_report/level1/tie_audit}"
TAXONOMY_PATH="${RGCN_TIE_AUDIT_TAXONOMY:-config/tie_taxonomy_ascribed_family_v1.json}"
SEEDS=(42 43 44)
MODELS=(rgcn rgat)

COMMON_ARGS=(
  --epochs 50
  --batch-size 512
  --num-neighbors 15,10
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
  --tie-taxonomy "$TAXONOMY_PATH"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_tie_audit_experiments.sh plan [all|matrix|diagnose|summarize]
  bash scripts/run_tie_audit_experiments.sh run  [all|matrix|diagnose|summarize]

Environment overrides:
  RGCN_PYTHON_BIN=.venv/bin/python
  RGCN_TIE_AUDIT_DATA=artifacts/level1_hierarchy/graph_data.pt
  RGCN_TIE_AUDIT_BASELINE_ROOT=runs_report/level1
  RGCN_TIE_AUDIT_OUTPUT_ROOT=runs_report/level1/tie_audit
  RGCN_TIE_AUDIT_TAXONOMY=config/tie_taxonomy_ascribed_family_v1.json

The script requires these completed legacy baselines and never retrains them:
  $RGCN_TIE_AUDIT_BASELINE_ROOT/rgcn_baseline/seed_{42,43,44}/metrics.json
  $RGCN_TIE_AUDIT_BASELINE_ROOT/rgat_baseline/seed_{42,43,44}/metrics.json

`matrix` runs only four conditions per model and seed: without_inherited,
random_matched_inherited, without_acquired, random_matched_acquired. `diagnose`
writes the static tie coverage/homophily report. `summarize` compares every
new condition with the existing baseline of the same model and seed. If both
paired runs retain test_predictions.csv, it also reports a paired test-node
bootstrap CI for the Macro-F1 difference.
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then
  usage
  exit 2
fi
if [[ ! " all matrix diagnose summarize " == *" $GROUP "* ]]; then
  usage
  exit 2
fi

selected() {
  [[ "$GROUP" == "all" || "$GROUP" == "$1" ]]
}

show_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

baseline_metrics_path() {
  local model="$1"
  local seed="$2"
  printf '%s/%s_baseline/seed_%s/metrics.json' "$BASELINE_ROOT" "$model" "$seed"
}

require_inputs() {
  if [[ "$MODE" != "run" ]]; then
    return
  fi
  if [[ ! -f "$DATA_PATH" ]]; then
    echo "Missing Level-1 artifact: $DATA_PATH" >&2
    exit 1
  fi
  if [[ ! -f "$TAXONOMY_PATH" ]]; then
    echo "Missing tie taxonomy: $TAXONOMY_PATH" >&2
    exit 1
  fi
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      local metrics_path
      metrics_path="$(baseline_metrics_path "$model" "$seed")"
      if [[ ! -f "$metrics_path" ]]; then
        echo "Required existing baseline is missing: $metrics_path" >&2
        echo "Refusing to train ablations without an exact same-seed baseline." >&2
        exit 1
      fi
    done
  done
}

show_baseline_reuse() {
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      printf '%-58s seed=%s  %s\n' \
        "[reuse; never retrain] ${model}_baseline" "$seed" "$(baseline_metrics_path "$model" "$seed")"
    done
  done
}

run_condition() {
  local model="$1"
  local condition="$2"
  shift 2
  local experiment="${model}__occupation_neighbours__${condition}"
  for seed in "${SEEDS[@]}"; do
    local output_dir="$OUTPUT_ROOT/$experiment/seed_$seed"
    local command=(
      "$PYTHON_BIN" run.py train
      --model "$model"
      --data "$DATA_PATH"
      --output-dir "$output_dir"
      "${COMMON_ARGS[@]}"
      --seed "$seed"
    )
    if [[ "$model" == "rgcn" ]]; then
      command+=(--rgcn-backend fast)
    fi
    command+=("$@")
    if [[ "$MODE" == "plan" ]]; then
      printf '%-58s seed=%s' "$experiment" "$seed"
      show_command "${command[@]}"
      continue
    fi
    if [[ -f "$output_dir/metrics.json" ]]; then
      echo "[skip] $experiment seed=$seed already has metrics.json"
      continue
    fi
    mkdir -p "$output_dir"
    printf '%q ' "${command[@]}" > "$output_dir/command.sh"
    printf '\n' >> "$output_dir/command.sh"
    echo "[run] $experiment seed=$seed"
    "${command[@]}"
  done
}

run_matrix() {
  for model in "${MODELS[@]}"; do
    run_condition "$model" without_inherited --drop-tie-groups inherited
    run_condition "$model" random_matched_inherited --match-random-drop-to-tie-groups inherited
    run_condition "$model" without_acquired --drop-tie-groups acquired
    run_condition "$model" random_matched_acquired --match-random-drop-to-tie-groups acquired
  done
}

require_inputs
if [[ "$MODE" == "plan" ]]; then
  show_baseline_reuse
fi
if selected matrix; then
  run_matrix
fi
if selected diagnose; then
  diagnostic_command=(
    "$PYTHON_BIN" run.py diagnose
    --data "$DATA_PATH"
    --output-dir "$OUTPUT_ROOT/diagnostics"
    --tie-taxonomy "$TAXONOMY_PATH"
    --homophily-label-split train
  )
  if [[ "$MODE" == "plan" ]]; then
    printf '%-58s' 'tie_audit_diagnostics'
    show_command "${diagnostic_command[@]}"
  else
    echo "[run] tie_audit_diagnostics"
    "${diagnostic_command[@]}"
  fi
fi
if selected summarize; then
  summary_command=(
    "$PYTHON_BIN" scripts/summarize_tie_audit_runs.py
    --root "$OUTPUT_ROOT"
    --baseline-root "$BASELINE_ROOT"
    --bootstrap-draws 2000
    --bootstrap-seed 20260813
  )
  if [[ "$MODE" == "plan" ]]; then
    printf '%-58s' 'tie_audit_paired_summary'
    show_command "${summary_command[@]}"
  else
    echo "[run] tie_audit_paired_summary"
    "${summary_command[@]}"
  fi
fi
