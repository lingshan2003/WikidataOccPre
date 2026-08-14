#!/usr/bin/env bash
# Independently prepared life-period inherited/acquired tie audit.
#
# This is deliberately separate from run_birth_cohort_tie_audit_experiments.sh.
# Every period first gets an induced graph: people whose known life spans
# overlap the period, plus people with only one known date in that date's
# period. Only edges whose two endpoints belong to the period remain. A person
# may therefore appear in multiple period artifacts. Each graph gets a fresh,
# fixed 70/10/20 split and its own `full` baseline.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
GROUP="${2:-all}"
PYTHON_BIN="${RGCN_PYTHON_BIN:-python}"
SOURCE_DATA="${RGCN_PERIOD_SOURCE_DATA:-artifacts/level1_hierarchy/graph_data.pt}"
ARTIFACT_ROOT="${RGCN_PERIOD_ARTIFACT_ROOT:-artifacts/level1_life_period_induced_v2}"
OUTPUT_ROOT="${RGCN_PERIOD_AUDIT_OUTPUT_ROOT:-runs_report/level1/life_period_induced_tie_audit_v2}"
TAXONOMY_PATH="${RGCN_TIE_AUDIT_TAXONOMY:-config/tie_taxonomy_ascribed_family_v1.json}"
LIFE_PERIOD_CONFIG="${RGCN_LIFE_PERIOD_CONFIG:-config/historical_life_periods_v2.json}"
SPLIT_SEED="${RGCN_PERIOD_SPLIT_SEED:-20260814}"
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
  bash scripts/run_period_induced_tie_audit_experiments.sh plan [all|prepare|matrix|summarize]
  bash scripts/run_period_induced_tie_audit_experiments.sh run  [all|prepare|matrix|summarize]

This is a period-contained experiment, not the earlier complete-graph local
intervention. It prepares one induced graph and fresh 70/10/20 split per life
period, then trains five NEW conditions for each model/seed:
  full, without_inherited, random_matched_inherited,
  without_acquired, random_matched_acquired.

Default matrix: 4 periods x 2 models x 5 conditions x 3 seeds = 120 runs.
Completed metrics.json files are skipped, so `run matrix` safely resumes.
It never reads or reuses runs_report/level1/{rgcn,rgat}_baseline.

Environment overrides:
  RGCN_PYTHON_BIN=.venv/bin/python
  RGCN_PERIOD_SOURCE_DATA=artifacts/level1_hierarchy/graph_data.pt
  RGCN_PERIOD_ARTIFACT_ROOT=artifacts/level1_life_period_induced_v2
  RGCN_PERIOD_AUDIT_OUTPUT_ROOT=runs_report/level1/life_period_induced_tie_audit_v2
  RGCN_LIFE_PERIOD_CONFIG=config/historical_life_periods_v2.json
  RGCN_PERIOD_SPLIT_SEED=20260814
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then
  usage
  exit 2
fi
if [[ ! " all prepare matrix summarize " == *" $GROUP "* ]]; then
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

load_periods() {
  "$PYTHON_BIN" -c '
from training.life_periods import load_life_period_config
import sys
for period in load_life_period_config(sys.argv[1]).periods:
    print(period.identifier)
' "$LIFE_PERIOD_CONFIG"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required input is missing: $path" >&2
    exit 1
  fi
}

require_preparation_inputs() {
  require_file "$SOURCE_DATA"
  require_file "$TAXONOMY_PATH"
  require_file "$LIFE_PERIOD_CONFIG"
}

require_prepared_artifacts() {
  local period
  require_file "$TAXONOMY_PATH"
  require_file "$LIFE_PERIOD_CONFIG"
  while IFS= read -r period; do
    [[ -z "$period" ]] && continue
    require_file "$ARTIFACT_ROOT/$period/graph_data.pt"
    require_file "$ARTIFACT_ROOT/$period/nodes.csv"
    require_file "$ARTIFACT_ROOT/$period/edges.csv"
    require_file "$ARTIFACT_ROOT/$period/split_summary.json"
  done < <(load_periods)
}

validate_matrix_artifacts() {
  # Refuse a costly matrix that contains no removable member of either tie
  # group in one period.  In particular, this catches a malformed artifact
  # before a direct ablation quietly becomes a no-op.
  local period
  while IFS= read -r period; do
    [[ -z "$period" ]] && continue
    "$PYTHON_BIN" -c '
import sys
import torch
from training.life_periods import load_life_period_config
from training.relation_controls import count_relation_pairs
from training.tie_taxonomy import load_tie_taxonomy, resolve_tie_ablation

data_path, taxonomy_path, cohort_path, period_id = sys.argv[1:]
bundle = torch.load(data_path, map_location="cpu", weights_only=False)
data, metadata = bundle["data"], bundle["metadata"]
details = metadata.get("period_induced_artifact", {})
config = load_life_period_config(cohort_path)
if details.get("edge_policy") != "retain_only_edges_with_both_endpoints_in_selected_life_period":
    raise ValueError("artifact is not an induced within-period graph")
if details.get("life_period_config", {}).get("sha256") != config.sha256:
    raise ValueError("artifact period configuration hash differs from requested config")
if details.get("selected_life_period", {}).get("id") != period_id:
    raise ValueError("artifact period ID differs from its directory/request")
taxonomy = load_tie_taxonomy(taxonomy_path, metadata["relation_to_id"])
all_pairs = count_relation_pairs(data, metadata["relation_to_id"].values(), metadata["relation_to_id"])
for group in ("inherited", "acquired"):
    relation_ids, _ = resolve_tie_ablation(taxonomy, (group,), metadata["relation_to_id"])
    pairs = count_relation_pairs(data, relation_ids, metadata["relation_to_id"])
    if pairs == 0:
        raise ValueError(f"{period_id}: no {group} relation pairs in the induced graph")
    if pairs > all_pairs:
        raise ValueError(f"{period_id}: {group} pair count exceeds random-control candidate set")
print(f"[validated] {period_id}: relation_pairs(all={all_pairs})")
' "$ARTIFACT_ROOT/$period/graph_data.pt" "$TAXONOMY_PATH" "$LIFE_PERIOD_CONFIG" "$period"
  done < <(load_periods)
}

prepare_artifacts() {
  local command=(
    "$PYTHON_BIN" scripts/prepare_life_period_induced_artifacts.py
    --source-data "$SOURCE_DATA"
    --output-root "$ARTIFACT_ROOT"
    --life-periods "$LIFE_PERIOD_CONFIG"
    --split-seed "$SPLIT_SEED"
  )
  if [[ "$MODE" == "plan" ]]; then
    printf '%-66s' 'prepare_period_induced_artifacts'
    show_command "${command[@]}"
  else
    echo "[run] prepare_period_induced_artifacts"
    "${command[@]}"
  fi
}

run_condition() {
  local model="$1"
  local period="$2"
  local condition="$3"
  shift 3
  local artifact="$ARTIFACT_ROOT/$period/graph_data.pt"
  local experiment="${model}__period_${period}__${condition}"
  local seed
  for seed in "${SEEDS[@]}"; do
    local output_dir="$OUTPUT_ROOT/$experiment/seed_$seed"
    local command=(
      "$PYTHON_BIN" run.py train
      --model "$model"
      --data "$artifact"
      --output-dir "$output_dir"
      "${COMMON_ARGS[@]}"
      --seed "$seed"
    )
    if [[ "$model" == "rgcn" ]]; then
      command+=(--rgcn-backend fast)
    fi
    command+=("$@")
    if [[ "$MODE" == "plan" ]]; then
      printf '%-66s seed=%s' "$experiment" "$seed"
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
  local period
  while IFS= read -r period; do
    [[ -z "$period" ]] && continue
    local model
    for model in "${MODELS[@]}"; do
      run_condition "$model" "$period" full
      run_condition "$model" "$period" without_inherited --drop-tie-groups inherited
      run_condition "$model" "$period" random_matched_inherited --match-random-drop-to-tie-groups inherited
      run_condition "$model" "$period" without_acquired --drop-tie-groups acquired
      run_condition "$model" "$period" random_matched_acquired --match-random-drop-to-tie-groups acquired
    done
  done < <(load_periods)
}

summarize() {
  local command=(
    "$PYTHON_BIN" scripts/summarize_period_induced_tie_audit.py
    --root "$OUTPUT_ROOT"
    --artifact-root "$ARTIFACT_ROOT"
    --life-periods "$LIFE_PERIOD_CONFIG"
    --tie-taxonomy "$TAXONOMY_PATH"
    --bootstrap-draws 2000
    --bootstrap-seed 20260814
  )
  if [[ "$MODE" == "plan" ]]; then
    printf '%-66s' 'period_induced_tie_audit_summary'
    show_command "${command[@]}"
  else
    echo "[run] period_induced_tie_audit_summary"
    "${command[@]}"
  fi
}

if selected prepare; then
  if [[ "$MODE" == "run" ]]; then
    require_preparation_inputs
  fi
  prepare_artifacts
fi
if selected matrix; then
  if [[ "$MODE" == "run" ]]; then
    require_prepared_artifacts
    validate_matrix_artifacts
  fi
  run_matrix
fi
if selected summarize; then
  if [[ "$MODE" == "run" ]]; then
    require_prepared_artifacts
  fi
  summarize
fi
