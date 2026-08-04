#!/usr/bin/env bash
# Build a clean, three-seed report matrix without mixing it into exploratory runs/.
# Intended for the CUDA server.  `plan` only prints commands; `run` executes them
# sequentially and records the exact command beside every metrics.json.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
GROUP="${2:-all}"
DATA_PATH="${RGCN_REPORT_DATA:-artifacts/level3_hierarchy/graph_data.pt}"
SEMANTIC_DATA_PATH="${RGCN_REPORT_SEMANTIC_DATA:-artifacts/level3_hierarchy/graph_data_semantic.pt}"
OUTPUT_ROOT="${RGCN_REPORT_OUTPUT_ROOT:-runs_report/level3}"
SEEDS=(42 43 44)

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
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_report_experiments.sh plan [all|architecture|features|relations|longtail|coverage]
  bash scripts/run_report_experiments.sh run  [all|architecture|features|relations|longtail|coverage]

Optional environment overrides:
  RGCN_REPORT_DATA=path/to/graph_data.pt
  RGCN_REPORT_SEMANTIC_DATA=path/to/graph_data_semantic.pt
  RGCN_REPORT_OUTPUT_ROOT=runs_report/level3

`run` skips a seed directory that already has metrics.json, making restart safe.
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then
  usage
  exit 2
fi

if [[ ! " all architecture features relations longtail coverage " == *" $GROUP "* ]]; then
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

run_condition() {
  local report_group="$1"
  local experiment="$2"
  local model="$3"
  local artifact="$4"
  shift 4

  if ! selected "$report_group"; then
    return
  fi
  if [[ "$MODE" == "run" && ! -f "$artifact" ]]; then
    echo "Missing artifact for $experiment: $artifact" >&2
    exit 1
  fi

  for seed in "${SEEDS[@]}"; do
    local output_dir="$OUTPUT_ROOT/$experiment/seed_$seed"
    local command=(
      python run.py train
      --model "$model"
      --data "$artifact"
      --output-dir "$output_dir"
      "${COMMON_ARGS[@]}"
      --seed "$seed"
      "$@"
    )
    if [[ "$MODE" == "plan" ]]; then
      printf '%-12s %-42s seed=%s' "$report_group" "$experiment" "$seed"
      show_command "${command[@]}"
      continue
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

# Group 1: does model architecture itself remove the Level-3 plateau?
run_condition architecture rgcn_baseline rgcn "$DATA_PATH" \
  --rgcn-backend fast --occupation-feature-levels 1,2,3 --auxiliary-features none
run_condition architecture rgat_baseline rgat "$DATA_PATH" \
  --occupation-feature-levels 1,2,3 --auxiliary-features none
run_condition architecture compgcn_baseline compgcn "$DATA_PATH" \
  --occupation-feature-levels 1,2,3 --auxiliary-features none --compgcn-composition mult

# Group 2: which input information changes performance?
run_condition features rgcn_structural rgcn "$DATA_PATH" \
  --rgcn-backend fast --feature-mode structural --occupation-feature-levels none --auxiliary-features none
run_condition features rgcn_l3_only rgcn "$DATA_PATH" \
  --rgcn-backend fast --occupation-feature-levels 3 --auxiliary-features none
run_condition features rgcn_hierarchy_with_aux rgcn "$DATA_PATH" \
  --rgcn-backend fast --occupation-feature-levels 1,2,3 --auxiliary-features country,temporal
run_condition features rgcn_semantic_occupation rgcn "$SEMANTIC_DATA_PATH" \
  --rgcn-backend fast --occupation-representation semantic --auxiliary-features none

# Group 3: separate relation semantics from graph-density effects.
run_condition relations rgcn_relation_shuffled rgcn "$DATA_PATH" \
  --rgcn-backend fast --occupation-feature-levels 1,2,3 --auxiliary-features none --shuffle-relation-types
run_condition relations rgcn_without_kinship rgcn "$DATA_PATH" \
  --rgcn-backend fast --occupation-feature-levels 1,2,3 --auxiliary-features none --drop-relation-groups kinship
run_condition relations rgcn_random_matched_kinship rgcn "$DATA_PATH" \
  --rgcn-backend fast --occupation-feature-levels 1,2,3 --auxiliary-features none --match-random-drop-to-relation-groups kinship
run_condition relations rgcn_without_education rgcn "$DATA_PATH" \
  --rgcn-backend fast --occupation-feature-levels 1,2,3 --auxiliary-features none --drop-relation-groups education_mentorship
run_condition relations rgcn_random_matched_education rgcn "$DATA_PATH" \
  --rgcn-backend fast --occupation-feature-levels 1,2,3 --auxiliary-features none --match-random-drop-to-relation-groups education_mentorship

# Group 4: do standard long-tail objectives improve the shared R-GCN baseline?
run_condition longtail rgcn_class_balanced rgcn "$DATA_PATH" \
  --rgcn-backend fast --occupation-feature-levels 1,2,3 --auxiliary-features none \
  --loss class_balanced --class-balanced-beta 0.9999
run_condition longtail rgcn_logit_adjusted rgcn "$DATA_PATH" \
  --rgcn-backend fast --occupation-feature-levels 1,2,3 --auxiliary-features none \
  --loss logit_adjusted --logit-adjustment-tau 1.0
run_condition longtail rgcn_balanced_roots rgcn "$DATA_PATH" \
  --rgcn-backend fast --occupation-feature-levels 1,2,3 --auxiliary-features none \
  --train-root-sampling class_balanced

# Group 5: can fuller neighbourhood access, rather than better modelling, help?
run_condition coverage compgcn_all_neighbours compgcn "$DATA_PATH" \
  --num-neighbors=-1,-1 --occupation-feature-levels 1,2,3 --auxiliary-features none --compgcn-composition mult
# These two use identical structural inputs and full-graph evaluation; only the
# optimisation mode changes. Full mode forbids occupation features by design.
run_condition coverage compgcn_structural_sampled_full_eval compgcn "$DATA_PATH" \
  --feature-mode structural --occupation-feature-levels none --auxiliary-features none \
  --eval-mode full --compgcn-composition mult
run_condition coverage compgcn_structural_full compgcn "$DATA_PATH" \
  --feature-mode structural --occupation-feature-levels none --auxiliary-features none \
  --train-mode full --eval-mode full --num-neighbors=-1,-1 --compgcn-composition mult

if [[ "$MODE" == "plan" ]]; then
  echo "Printed 18 conditions × 3 seeds = 54 training commands."
  echo "Semantic condition requires: $SEMANTIC_DATA_PATH"
else
  echo "Completed selected report group '$GROUP'."
  echo "Summarise with: python scripts/summarize_report_runs.py --root $OUTPUT_ROOT"
fi
