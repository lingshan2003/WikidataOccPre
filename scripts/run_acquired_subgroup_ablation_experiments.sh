#!/usr/bin/env bash
# R-GCN-only complete-graph ablations for the subgroups inside acquired ties.
#
# This reuses the completed Level-1 R-GCN full-graph baselines.  Every direct
# deletion is paired with a same-seed control that removes exactly the same
# number of original-edge/generated-reverse units from the whole graph.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
GROUP="${2:-all}"
PYTHON_BIN="${RGCN_PYTHON_BIN:-python}"
DATA_PATH="${RGCN_ACQUIRED_SUBGROUP_DATA:-artifacts/level1_hierarchy/graph_data.pt}"
BASELINE_ROOT="${RGCN_ACQUIRED_SUBGROUP_BASELINE_ROOT:-runs_report/level1}"
OUTPUT_ROOT="${RGCN_ACQUIRED_SUBGROUP_OUTPUT_ROOT:-runs_report/level1/acquired_tie_subgroups_v1}"
TIE_TAXONOMY_PATH="${RGCN_TIE_AUDIT_TAXONOMY:-config/tie_taxonomy_ascribed_family_v1.json}"
RELATION_TAXONOMY_PATH="${RGCN_ACQUIRED_SUBGROUP_TAXONOMY:-config/tie_taxonomy_acquired_subgroups_v1.json}"
EXPECTED_TARGET_COLUMN="${RGCN_ACQUIRED_SUBGROUP_EXPECT_TARGET_COLUMN:-occupation_level1}"
# The residual group is deliberately optional: its labels are heterogeneous
# and sparse, so it is not a pre-registered primary comparison across periods.
ACQUIRED_GROUPS="${RGCN_ACQUIRED_SUBGROUPS:-intimate_partnership,education_mentorship,professional_collaboration,influence_succession,religious_ordination}"
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
  --occupation-feature-levels 1,2,3
  --auxiliary-features none
  --tie-taxonomy "$TIE_TAXONOMY_PATH"
  --relation-taxonomy "$RELATION_TAXONOMY_PATH"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_acquired_subgroup_ablation_experiments.sh plan [all|matrix|summarize]
  bash scripts/run_acquired_subgroup_ablation_experiments.sh run  [all|matrix|summarize]

R-GCN only. The default primary groups are intimate partnership, education /
mentorship, professional collaboration, influence / succession, and religious
ordination. It reuses (and never retrains) the three existing Level-1 R-GCN
full-graph baselines. The matrix has 5 groups x 2 conditions x 3 seeds = 30
new runs. Every random control deletes the exact directed-edge count of its
matching subgroup, preserving each original edge and generated reverse edge as
one unit.

Environment overrides:
  RGCN_PYTHON_BIN=.venv/bin/python
  RGCN_ACQUIRED_SUBGROUP_DATA=artifacts/level1_hierarchy/graph_data.pt
  RGCN_ACQUIRED_SUBGROUP_BASELINE_ROOT=runs_report/level1
  RGCN_ACQUIRED_SUBGROUP_OUTPUT_ROOT=runs_report/level1/acquired_tie_subgroups_v1
  RGCN_ACQUIRED_SUBGROUP_TAXONOMY=config/tie_taxonomy_acquired_subgroups_v1.json
  RGCN_ACQUIRED_SUBGROUP_EXPECT_TARGET_COLUMN=occupation_level1
  RGCN_ACQUIRED_SUBGROUPS=intimate_partnership,...,other_acquired
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then usage; exit 2; fi
if [[ ! " all matrix summarize " == *" $GROUP "* ]]; then usage; exit 2; fi

selected() { [[ "$GROUP" == "all" || "$GROUP" == "$1" ]]; }
show_command() { printf ' '; printf '%q ' "$@"; printf '\n'; }

baseline_metrics_path() {
  local seed="$1"
  printf '%s/rgcn_baseline/seed_%s/metrics.json' "$BASELINE_ROOT" "$seed"
}

validate_inputs() {
  [[ -f "$DATA_PATH" ]] || { echo "Missing Level-1 artifact: $DATA_PATH" >&2; exit 1; }
  [[ -f "$TIE_TAXONOMY_PATH" ]] || { echo "Missing tie taxonomy: $TIE_TAXONOMY_PATH" >&2; exit 1; }
  [[ -f "$RELATION_TAXONOMY_PATH" ]] || { echo "Missing relation taxonomy: $RELATION_TAXONOMY_PATH" >&2; exit 1; }
  local seed
  for seed in "${SEEDS[@]}"; do
    [[ -f "$(baseline_metrics_path "$seed")" ]] || {
      echo "Required existing R-GCN baseline is missing: $(baseline_metrics_path "$seed")" >&2
      exit 1
    }
  done
  "$PYTHON_BIN" -c '
import sys
import torch
from training.relation_controls import count_edge_instance_pairs
from training.tie_taxonomy import load_relation_taxonomy, resolve_relation_taxonomy_ablation

data_path, taxonomy_path, selected, expected_target = sys.argv[1:]
bundle = torch.load(data_path, map_location="cpu", weights_only=False)
if expected_target and bundle["metadata"].get("target_column") != expected_target:
    raise ValueError(
        f"target_column={bundle['metadata'].get('target_column')!r}; expected {expected_target!r}"
    )
taxonomy = load_relation_taxonomy(taxonomy_path, bundle["metadata"]["relation_to_id"])
groups = tuple(item for item in selected.split(",") if item)
if not groups:
    raise ValueError("At least one acquired subgroup is required")
if "inherited" in groups:
    raise ValueError("This runner is for acquired subgroups; inherited belongs to the prior two-way audit")
for group in groups:
    relation_ids, _ = resolve_relation_taxonomy_ablation(taxonomy, (group,), bundle["metadata"]["relation_to_id"])
    count = count_edge_instance_pairs(bundle["data"], relation_ids, bundle["metadata"]["relation_to_id"])
    if count == 0:
        raise ValueError(f"No edge instances for acquired subgroup {group!r}; refusing a no-op ablation")
    print(f"[validated] {group}: edge_instance_pairs={count}")
' "$DATA_PATH" "$RELATION_TAXONOMY_PATH" "$ACQUIRED_GROUPS" "$EXPECTED_TARGET_COLUMN"
}

run_condition() {
  local subgroup="$1"
  local condition="$2"
  shift 2
  local experiment="rgcn__occupation_neighbours__${condition}"
  local seed
  for seed in "${SEEDS[@]}"; do
    local output_dir="$OUTPUT_ROOT/$experiment/seed_$seed"
    local command=(
      "$PYTHON_BIN" run.py train
      --model rgcn
      --rgcn-backend fast
      --data "$DATA_PATH"
      --output-dir "$output_dir"
      "${COMMON_ARGS[@]}"
      --seed "$seed"
      "$@"
    )
    if [[ "$MODE" == "plan" ]]; then
      printf '%-76s seed=%s' "$experiment" "$seed"
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
  local subgroup
  IFS=',' read -r -a subgroups <<< "$ACQUIRED_GROUPS"
  for subgroup in "${subgroups[@]}"; do
    [[ -n "$subgroup" ]] || continue
    run_condition "$subgroup" "without_${subgroup}" --drop-relation-taxonomy-groups "$subgroup"
    run_condition "$subgroup" "random_matched_${subgroup}" --match-random-drop-to-relation-taxonomy-groups "$subgroup"
  done
}

summarize() {
  local command=(
    "$PYTHON_BIN" scripts/summarize_acquired_subgroup_runs.py
    --scope full
    --root "$OUTPUT_ROOT"
    --baseline-root "$BASELINE_ROOT"
    --relation-taxonomy "$RELATION_TAXONOMY_PATH"
    --groups "$ACQUIRED_GROUPS"
    --bootstrap-draws "${RGCN_ACQUIRED_SUBGROUP_BOOTSTRAP_DRAWS:-2000}"
    --bootstrap-seed 20260817
  )
  if [[ "$MODE" == "plan" ]]; then
    printf '%-76s' 'acquired_subgroup_full_graph_summary'
    show_command "${command[@]}"
  else
    echo "[run] acquired_subgroup_full_graph_summary"
    "${command[@]}"
  fi
}

if [[ "$MODE" == "run" ]]; then validate_inputs; fi
if selected matrix; then run_matrix; fi
if selected summarize; then summarize; fi
