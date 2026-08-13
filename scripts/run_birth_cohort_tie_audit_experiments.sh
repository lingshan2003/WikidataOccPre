#!/usr/bin/env bash
# Birth-cohort-specific inherited/acquired relation audit.
#
# The existing complete-graph Level-1 baselines are reused.  For each editable
# birth cohort, this runs four cohort-incident interventions (two tie groups
# and their equal-size cohort-incident random-pair controls) across R-GCN and
# R-GAT, seeds 42/43/44: 4 cohorts x 2 models x 4 conditions x 3 seeds = 96
# new runs with the default configuration.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
GROUP="${2:-all}"
PYTHON_BIN="${RGCN_PYTHON_BIN:-python}"
DATA_PATH="${RGCN_BIRTH_COHORT_DATA:-artifacts/level1_hierarchy/graph_data.pt}"
BASELINE_ROOT="${RGCN_BIRTH_COHORT_BASELINE_ROOT:-runs_report/level1}"
GLOBAL_AUDIT_ROOT="${RGCN_GLOBAL_TIE_AUDIT_ROOT:-runs_report/level1/tie_audit}"
OUTPUT_ROOT="${RGCN_BIRTH_COHORT_OUTPUT_ROOT:-runs_report/level1/birth_cohort_tie_audit}"
TAXONOMY_PATH="${RGCN_TIE_AUDIT_TAXONOMY:-config/tie_taxonomy_ascribed_family_v1.json}"
COHORT_CONFIG="${RGCN_BIRTH_COHORT_CONFIG:-config/birth_cohorts_historical_eras_v1.json}"
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
  --edge-cohort-config "$COHORT_CONFIG"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_birth_cohort_tie_audit_experiments.sh plan [all|matrix|summarize-global|summarize-targeted]
  bash scripts/run_birth_cohort_tie_audit_experiments.sh run  [all|matrix|summarize-global|summarize-targeted]

This script never reruns the six complete-graph baselines.  For each cohort it
deletes only relation pairs incident to people born in that cohort.  Its random
control deletes the same number of random relation pairs from the same
cohort-incident candidate set, controlling both edge quantity and exposure.

Environment overrides:
  RGCN_PYTHON_BIN=.venv/bin/python
  RGCN_BIRTH_COHORT_DATA=artifacts/level1_hierarchy/graph_data.pt
  RGCN_BIRTH_COHORT_BASELINE_ROOT=runs_report/level1
  RGCN_GLOBAL_TIE_AUDIT_ROOT=runs_report/level1/tie_audit
  RGCN_BIRTH_COHORT_OUTPUT_ROOT=runs_report/level1/birth_cohort_tie_audit
  RGCN_BIRTH_COHORT_CONFIG=config/birth_cohorts_historical_eras_v1.json
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then
  usage
  exit 2
fi
if [[ ! " all matrix summarize-global summarize-targeted " == *" $GROUP "* ]]; then
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
  for required in "$DATA_PATH" "$TAXONOMY_PATH" "$COHORT_CONFIG"; do
    if [[ ! -f "$required" ]]; then
      echo "Required input is missing: $required" >&2
      exit 1
    fi
  done
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      local metrics_path
      metrics_path="$(baseline_metrics_path "$model" "$seed")"
      if [[ ! -f "$metrics_path" ]]; then
        echo "Required baseline is missing: $metrics_path" >&2
        exit 1
      fi
    done
  done
}

load_cohorts() {
  "$PYTHON_BIN" -c '
from training.birth_cohorts import load_birth_cohort_config
import sys
for cohort in load_birth_cohort_config(sys.argv[1]).cohorts:
    print(cohort.identifier)
' "$COHORT_CONFIG"
}

show_baseline_reuse() {
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      printf '%-65s seed=%s  %s\n' \
        "[reuse; never retrain] ${model}_baseline" "$seed" "$(baseline_metrics_path "$model" "$seed")"
    done
  done
}

run_condition() {
  local model="$1"
  local cohort="$2"
  local condition="$3"
  shift 3
  local experiment="${model}__cohort_${cohort}__${condition}"
  for seed in "${SEEDS[@]}"; do
    local output_dir="$OUTPUT_ROOT/$experiment/seed_$seed"
    local command=(
      "$PYTHON_BIN" run.py train
      --model "$model"
      --data "$DATA_PATH"
      --output-dir "$output_dir"
      "${COMMON_ARGS[@]}"
      --edge-cohort-id "$cohort"
      --seed "$seed"
    )
    if [[ "$model" == "rgcn" ]]; then
      command+=(--rgcn-backend fast)
    fi
    command+=("$@")
    if [[ "$MODE" == "plan" ]]; then
      printf '%-65s seed=%s' "$experiment" "$seed"
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
  local cohorts
  cohorts="$(load_cohorts)"
  while IFS= read -r cohort; do
    [[ -z "$cohort" ]] && continue
    for model in "${MODELS[@]}"; do
      run_condition "$model" "$cohort" without_inherited --drop-tie-groups inherited
      run_condition "$model" "$cohort" random_matched_inherited --match-random-drop-to-tie-groups inherited
      run_condition "$model" "$cohort" without_acquired --drop-tie-groups acquired
      run_condition "$model" "$cohort" random_matched_acquired --match-random-drop-to-tie-groups acquired
    done
  done <<< "$cohorts"
}

require_inputs
if [[ "$MODE" == "plan" ]]; then
  show_baseline_reuse
fi
if selected matrix; then
  run_matrix
fi
if selected summarize-global; then
  command=(
    "$PYTHON_BIN" scripts/summarize_tie_audit_birth_cohorts.py
    --root "$GLOBAL_AUDIT_ROOT"
    --baseline-root "$BASELINE_ROOT"
    --data "$DATA_PATH"
    --birth-cohorts "$COHORT_CONFIG"
  )
  if [[ "$MODE" == "plan" ]]; then
    printf '%-65s' 'birth_cohort_global_prediction_summary'
    show_command "${command[@]}"
  else
    echo "[run] birth_cohort_global_prediction_summary"
    "${command[@]}"
  fi
fi
if selected summarize-targeted; then
  command=(
    "$PYTHON_BIN" scripts/summarize_tie_audit_birth_cohorts.py
    --root "$GLOBAL_AUDIT_ROOT"
    --targeted-root "$OUTPUT_ROOT"
    --baseline-root "$BASELINE_ROOT"
    --data "$DATA_PATH"
    --birth-cohorts "$COHORT_CONFIG"
  )
  if [[ "$MODE" == "plan" ]]; then
    printf '%-65s' 'birth_cohort_targeted_intervention_summary'
    show_command "${command[@]}"
  else
    echo "[run] birth_cohort_targeted_intervention_summary"
    "${command[@]}"
  fi
fi
