#!/usr/bin/env bash
# R-GCN-only acquired-tie subgroup ablations in independently induced periods.
#
# The script intentionally reuses the already-trained R-GCN `full` baseline
# for each period. It does not reinterpret the prior complete-graph local
# cohort intervention as a period-contained result.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
GROUP="${2:-all}"
PYTHON_BIN="${RGCN_PYTHON_BIN:-python}"
ARTIFACT_ROOT="${RGCN_PERIOD_ARTIFACT_ROOT:-artifacts/level1_life_period_induced_v2}"
BASELINE_ROOT="${RGCN_PERIOD_SUBGROUP_BASELINE_ROOT:-runs_report/level1/life_period_induced_tie_audit_v2}"
OUTPUT_ROOT="${RGCN_PERIOD_SUBGROUP_OUTPUT_ROOT:-runs_report/level1/life_period_induced_acquired_tie_subgroups_v1}"
TIE_TAXONOMY_PATH="${RGCN_TIE_AUDIT_TAXONOMY:-config/tie_taxonomy_ascribed_family_v1.json}"
RELATION_TAXONOMY_PATH="${RGCN_ACQUIRED_SUBGROUP_TAXONOMY:-config/tie_taxonomy_acquired_subgroups_v1.json}"
LIFE_PERIOD_CONFIG="${RGCN_LIFE_PERIOD_CONFIG:-config/historical_life_periods_v2.json}"
ACQUIRED_GROUPS="${RGCN_ACQUIRED_SUBGROUPS:-intimate_partnership,education_mentorship,professional_collaboration,influence_succession,religious_ordination}"
OCCUPATION_FEATURE_LEVELS="${RGCN_PERIOD_OCCUPATION_FEATURE_LEVELS:-1,2,3}"
EXPECTED_TARGET_COLUMN="${RGCN_PERIOD_EXPECT_TARGET_COLUMN:-occupation_level1}"
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
  --occupation-feature-levels "$OCCUPATION_FEATURE_LEVELS"
  --auxiliary-features none
  --tie-taxonomy "$TIE_TAXONOMY_PATH"
  --relation-taxonomy "$RELATION_TAXONOMY_PATH"
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_period_induced_acquired_subgroup_experiments.sh plan [all|matrix|summarize]
  bash scripts/run_period_induced_acquired_subgroup_experiments.sh run  [all|matrix|summarize]

R-GCN only. This is a period-contained follow-up to the two-way tie audit:
it reuses each period's independently induced graph and its completed same-seed
R-GCN full baseline. It trains 5 acquired subgroups x (direct + exact matched
random) x 4 periods x 3 seeds = 120 new runs by default. It never uses the
old complete-graph cohort-local results as a period baseline.

The residual other_acquired group is excluded from the default primary matrix
because it is sparse and semantically heterogeneous. Add it explicitly with
RGCN_ACQUIRED_SUBGROUPS if an exploratory residual check is wanted.
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then usage; exit 2; fi
if [[ ! " all matrix summarize " == *" $GROUP "* ]]; then usage; exit 2; fi

selected() { [[ "$GROUP" == "all" || "$GROUP" == "$1" ]]; }
show_command() { printf ' '; printf '%q ' "$@"; printf '\n'; }

load_periods() {
  "$PYTHON_BIN" -c '
from training.life_periods import load_life_period_config
import sys
for period in load_life_period_config(sys.argv[1]).periods:
    print(period.identifier)
' "$LIFE_PERIOD_CONFIG"
}

baseline_metrics_path() {
  local period="$1"
  local seed="$2"
  printf '%s/rgcn__period_%s__full/seed_%s/metrics.json' "$BASELINE_ROOT" "$period" "$seed"
}

validate_inputs() {
  [[ -f "$TIE_TAXONOMY_PATH" ]] || { echo "Missing tie taxonomy: $TIE_TAXONOMY_PATH" >&2; exit 1; }
  [[ -f "$RELATION_TAXONOMY_PATH" ]] || { echo "Missing relation taxonomy: $RELATION_TAXONOMY_PATH" >&2; exit 1; }
  [[ -f "$LIFE_PERIOD_CONFIG" ]] || { echo "Missing life-period config: $LIFE_PERIOD_CONFIG" >&2; exit 1; }
  local period seed artifact
  while IFS= read -r period; do
    [[ -n "$period" ]] || continue
    artifact="$ARTIFACT_ROOT/$period/graph_data.pt"
    [[ -f "$artifact" ]] || { echo "Missing induced period artifact: $artifact" >&2; exit 1; }
    for seed in "${SEEDS[@]}"; do
      [[ -f "$(baseline_metrics_path "$period" "$seed")" ]] || {
        echo "Missing required period R-GCN full baseline: $(baseline_metrics_path "$period" "$seed")" >&2
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
metadata = bundle["metadata"]
if expected_target and metadata.get("target_column") != expected_target:
    raise ValueError(f"target_column={metadata.get('target_column')!r}; expected {expected_target!r}")
details = metadata.get("period_induced_artifact", {})
if details.get("edge_policy") != "retain_only_edges_with_both_endpoints_in_selected_life_period":
    raise ValueError("artifact is not an induced within-period graph")
taxonomy = load_relation_taxonomy(taxonomy_path, metadata["relation_to_id"])
for group in tuple(item for item in selected.split(",") if item):
    if group == "inherited":
        raise ValueError("This runner is for acquired subgroups; inherited belongs to the prior audit")
    relation_ids, _ = resolve_relation_taxonomy_ablation(taxonomy, (group,), metadata["relation_to_id"])
    count = count_edge_instance_pairs(bundle["data"], relation_ids, metadata["relation_to_id"])
    if count == 0:
        raise ValueError(f"No edge instances for acquired subgroup {group!r}; refusing a no-op ablation")
    print(f"[validated] {group}: edge_instance_pairs={count}")
' "$artifact" "$RELATION_TAXONOMY_PATH" "$ACQUIRED_GROUPS" "$EXPECTED_TARGET_COLUMN"
  done < <(load_periods)
}

run_condition() {
  local period="$1"
  local condition="$2"
  shift 2
  local artifact="$ARTIFACT_ROOT/$period/graph_data.pt"
  local experiment="rgcn__period_${period}__${condition}"
  local seed
  for seed in "${SEEDS[@]}"; do
    local output_dir="$OUTPUT_ROOT/$experiment/seed_$seed"
    local command=(
      "$PYTHON_BIN" run.py train
      --model rgcn
      --rgcn-backend fast
      --data "$artifact"
      --output-dir "$output_dir"
      "${COMMON_ARGS[@]}"
      --seed "$seed"
      "$@"
    )
    if [[ "$MODE" == "plan" ]]; then
      printf '%-84s seed=%s' "$experiment" "$seed"
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
  local period subgroup
  local subgroups=()
  IFS=',' read -r -a subgroups <<< "$ACQUIRED_GROUPS"
  while IFS= read -r period; do
    [[ -n "$period" ]] || continue
    for subgroup in "${subgroups[@]}"; do
      [[ -n "$subgroup" ]] || continue
      run_condition "$period" "without_${subgroup}" --drop-relation-taxonomy-groups "$subgroup"
      run_condition "$period" "random_matched_${subgroup}" --match-random-drop-to-relation-taxonomy-groups "$subgroup"
    done
  done < <(load_periods)
}

summarize() {
  local command=(
    "$PYTHON_BIN" scripts/summarize_acquired_subgroup_runs.py
    --scope period
    --root "$OUTPUT_ROOT"
    --baseline-root "$BASELINE_ROOT"
    --relation-taxonomy "$RELATION_TAXONOMY_PATH"
    --groups "$ACQUIRED_GROUPS"
    --life-periods "$LIFE_PERIOD_CONFIG"
    --bootstrap-draws "${RGCN_ACQUIRED_SUBGROUP_BOOTSTRAP_DRAWS:-2000}"
    --bootstrap-seed 20260817
  )
  if [[ "$MODE" == "plan" ]]; then
    printf '%-84s' 'period_induced_acquired_subgroup_summary'
    show_command "${command[@]}"
  else
    echo "[run] period_induced_acquired_subgroup_summary"
    "${command[@]}"
  fi
}

if [[ "$MODE" == "run" ]]; then validate_inputs; fi
if selected matrix; then run_matrix; fi
if selected summarize; then summarize; fi
