#!/usr/bin/env bash
# Train two-layer L1 R-GATs and GraphMask probes for three relation vocabularies
# in every induced historical life-period graph.
#
# Each period gets three independent models:
#   - binary:     inherited_ties / acquired_ties
#   - multi_group: inherited plus the acquired-tie subgroups
#   - exact:      the original exact directed relation vocabulary
#
# The script deliberately does not reuse one-layer period checkpoints.  It is
# restart-safe: compatible period/collapsed artifacts, checkpoints, probes and
# complete reports are reused; incompatible existing outputs cause a failure
# rather than being overwritten silently.

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
STAGE="${2:-all}"

PYTHON_BIN="${RGCN_PYTHON_BIN:-python}"
SOURCE_DATA="${RGCN_PERIOD_SOURCE_DATA:-artifacts/level1_hierarchy/graph_data.pt}"
PERIOD_ARTIFACT_ROOT="${RGCN_PERIOD_ARTIFACT_ROOT:-artifacts/level1_life_period_induced_v2}"
RELATION_ARTIFACT_ROOT="${RGCN_PERIOD_RELATION_ARTIFACT_ROOT:-artifacts/level1_life_period_relation_representations_v1}"
MODEL_ROOT="${RGCN_PERIOD_RELATION_MODEL_ROOT:-runs_report/level1/life_period_relation_representations_rgat_v1}"
GRAPHMASK_ROOT="${RGCN_PERIOD_RELATION_GRAPHMASK_ROOT:-runs_graphmask/level1/life_period_relation_representations_rgat_v1}"

LIFE_PERIOD_CONFIG="${RGCN_LIFE_PERIOD_CONFIG:-config/historical_life_periods_v2.json}"
TIE_TAXONOMY_PATH="${RGCN_TIE_AUDIT_TAXONOMY:-config/tie_taxonomy_ascribed_family_v1.json}"
MULTI_GROUP_TAXONOMY_PATH="${RGCN_ACQUIRED_SUBGROUP_TAXONOMY:-config/tie_taxonomy_acquired_subgroups_v1.json}"
SPLIT_SEED="${RGCN_PERIOD_SPLIT_SEED:-20260814}"

MODEL_SEED="${RGCN_PERIOD_RELATION_SEED:-42}"
DEVICE="${RGCN_PERIOD_RELATION_DEVICE:-cuda:0}"
TRAIN_NUM_NEIGHBORS="${RGCN_PERIOD_RELATION_NUM_NEIGHBORS:-15,10}"
TRAIN_BATCH_SIZE="${RGCN_PERIOD_RELATION_BATCH_SIZE:-512}"
TRAIN_NUM_WORKERS="${RGCN_PERIOD_RELATION_NUM_WORKERS:-4}"
GRAPHMASK_NUM_NEIGHBORS="${RGCN_PERIOD_GRAPHMASK_NUM_NEIGHBORS:-full}"
GRAPHMASK_BATCH_SIZE="${RGCN_PERIOD_GRAPHMASK_BATCH_SIZE:-32}"
# Zero avoids PyTorch worker-cleanup warnings and makes GraphMask validation
# graphs deterministic. Override to 4 only after confirming the server's
# DataLoader workers exit cleanly.
GRAPHMASK_NUM_WORKERS="${RGCN_PERIOD_GRAPHMASK_NUM_WORKERS:-0}"
GRAPHMASK_EPOCHS_PER_LAYER="${RGCN_PERIOD_GRAPHMASK_EPOCHS_PER_LAYER:-3}"
GRAPHMASK_TOP_K="${RGCN_PERIOD_GRAPHMASK_TOP_K:-50}"
OCCUPATION_FEATURE_LEVELS="${RGCN_PERIOD_OCCUPATION_FEATURE_LEVELS:-1,2,3}"

REPRESENTATIONS=(binary multi_group exact)
REPORT_FILES=(
  test_metrics.json
  relations_directed.csv
  relations_base.csv
  root_top_edges.csv.gz
  manifest.json
)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_period_relation_graphmask.sh plan [all|prepare|collapse|train|graphmask]
  bash scripts/run_period_relation_graphmask.sh run  [all|prepare|collapse|train|graphmask]

The default matrix is one seed, four historical life periods, and three
independently trained two-layer R-GAT relation representations (12 models),
followed by one GraphMask probe and relation report for each checkpoint.

Stages:
  prepare    build/reuse the four within-period induced graph artifacts
  collapse   build/reuse binary and multi-group artifacts from each period graph
  train      train/reuse the 12 two-layer R-GAT checkpoints
  graphmask  train/reuse probes and export the 12 relation reports
  all        run the stages above in their required order

Important defaults:
  * model training is sampled two-hop R-GAT with --num-neighbors 15,10;
  * GraphMask uses full two-hop neighbourhoods for comparability with the
    existing L1 R-GAT GraphMask analysis;
  * only seed 42 is used.  Set RGCN_PERIOD_RELATION_SEED to change it.

Useful overrides:
  RGCN_PYTHON_BIN=.venv/bin/python
  RGCN_PERIOD_RELATION_DEVICE=cuda:0
  RGCN_PERIOD_GRAPHMASK_NUM_NEIGHBORS=auto  # use checkpoint fan-outs instead
  RGCN_PERIOD_RELATION_ARTIFACT_ROOT=artifacts/my_period_relation_artifacts
  RGCN_PERIOD_RELATION_MODEL_ROOT=runs_report/level1/my_period_models
  RGCN_PERIOD_RELATION_GRAPHMASK_ROOT=runs_graphmask/level1/my_period_graphmask
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then
  usage
  exit 2
fi
if [[ ! " all prepare collapse train graphmask " == *" $STAGE "* ]]; then
  usage
  exit 2
fi

selected() {
  [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]
}

show_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "Required input is missing: $1"
}

load_periods() {
  "$PYTHON_BIN" -c '
from training.life_periods import load_life_period_config
import sys
for period in load_life_period_config(sys.argv[1]).periods:
    print(period.identifier)
' "$LIFE_PERIOD_CONFIG"
}

period_graph_path() {
  local period="$1"
  printf '%s/%s/graph_data.pt' "$PERIOD_ARTIFACT_ROOT" "$period"
}

representation_dir() {
  local period="$1"
  local representation="$2"
  printf '%s/%s/%s' "$RELATION_ARTIFACT_ROOT" "$period" "$representation"
}

artifact_path() {
  local period="$1"
  local representation="$2"
  case "$representation" in
    exact) period_graph_path "$period" ;;
    binary|multi_group) printf '%s/graph_data.pt' "$(representation_dir "$period" "$representation")" ;;
    *) die "Unknown representation: $representation" ;;
  esac
}

tie_taxonomy_for_representation() {
  local period="$1"
  local representation="$2"
  case "$representation" in
    exact) printf '%s' "$TIE_TAXONOMY_PATH" ;;
    binary) printf '%s/binary_tie_taxonomy.json' "$(representation_dir "$period" binary)" ;;
    multi_group) printf '%s/collapsed_tie_taxonomy.json' "$(representation_dir "$period" multi_group)" ;;
    *) die "Unknown representation: $representation" ;;
  esac
}

model_dir() {
  local period="$1"
  local representation="$2"
  printf '%s/rgat__period_%s__%s/seed_%s' \
    "$MODEL_ROOT" "$period" "$representation" "$MODEL_SEED"
}

checkpoint_path() {
  local period="$1"
  local representation="$2"
  printf '%s/best_model.pt' "$(model_dir "$period" "$representation")"
}

graphmask_dir() {
  local period="$1"
  local representation="$2"
  printf '%s/rgat__period_%s__%s/seed_%s' \
    "$GRAPHMASK_ROOT" "$period" "$representation" "$MODEL_SEED"
}

probe_path() {
  local period="$1"
  local representation="$2"
  printf '%s/graphmask_probe.pt' "$(graphmask_dir "$period" "$representation")"
}

report_dir() {
  local period="$1"
  local representation="$2"
  printf '%s/test_report' "$(graphmask_dir "$period" "$representation")"
}

require_base_inputs() {
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] \
    || die "Python executable is unavailable: $PYTHON_BIN"
  require_file "$LIFE_PERIOD_CONFIG"
  require_file "$TIE_TAXONOMY_PATH"
  require_file "$MULTI_GROUP_TAXONOMY_PATH"
}

require_period_artifacts() {
  local period
  while IFS= read -r period; do
    [[ -n "$period" ]] || continue
    require_file "$(period_graph_path "$period")"
    require_file "$PERIOD_ARTIFACT_ROOT/$period/nodes.csv"
    require_file "$PERIOD_ARTIFACT_ROOT/$period/edges.csv"
  done < <(load_periods)
}

require_relation_artifacts() {
  local period representation
  while IFS= read -r period; do
    [[ -n "$period" ]] || continue
    for representation in "${REPRESENTATIONS[@]}"; do
      require_file "$(artifact_path "$period" "$representation")"
      require_file "$(tie_taxonomy_for_representation "$period" "$representation")"
    done
  done < <(load_periods)
}

collapsed_artifact_is_compatible() {
  local operation="$1"
  local source_data="$2"
  local taxonomy="$3"
  local output_dir="$4"
  "$PYTHON_BIN" - "$operation" "$source_data" "$taxonomy" "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

from training.tie_taxonomy import sha256_file

operation, source_data, taxonomy, output_dir = sys.argv[1:]
source_data = Path(source_data)
taxonomy = Path(taxonomy)
output_dir = Path(output_dir)
manifest_path = output_dir / "relation_collapse_manifest.json"
artifact_path = output_dir / "graph_data.pt"
if not artifact_path.is_file() or not manifest_path.is_file():
    raise SystemExit(1)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("operation") != operation:
    raise SystemExit(1)
if manifest.get("source_data_sha256") != sha256_file(source_data):
    raise SystemExit(1)
if manifest.get("source_taxonomy", {}).get("sha256") != sha256_file(taxonomy):
    raise SystemExit(1)
PY
}

collapse_artifact() {
  local representation="$1"
  local period="$2"
  local source_data output_dir taxonomy operation command_name
  source_data="$(period_graph_path "$period")"
  output_dir="$(representation_dir "$period" "$representation")"
  case "$representation" in
    binary)
      taxonomy="$TIE_TAXONOMY_PATH"
      operation="collapse_base_relations_to_inherited_acquired_ties"
      command_name="collapse-ties"
      ;;
    multi_group)
      taxonomy="$MULTI_GROUP_TAXONOMY_PATH"
      operation="collapse_base_relations_to_taxonomy_groups"
      command_name="collapse-relations"
      ;;
    *) die "Only binary and multi_group require relation collapse" ;;
  esac

  if [[ -e "$output_dir/graph_data.pt" || -e "$output_dir/relation_collapse_manifest.json" ]]; then
    if collapsed_artifact_is_compatible "$operation" "$source_data" "$taxonomy" "$output_dir"; then
      echo "[skip collapse] period=$period representation=$representation"
      return
    fi
    die "Existing collapsed artifact is incomplete or incompatible: $output_dir. Choose a new RGCN_PERIOD_RELATION_ARTIFACT_ROOT; do not overwrite it."
  fi

  mkdir -p "$output_dir"
  local command=(
    "$PYTHON_BIN" run.py "$command_name"
    --data "$source_data"
    --output-dir "$output_dir"
  )
  if [[ "$representation" == "binary" ]]; then
    command+=(--tie-taxonomy "$taxonomy")
  else
    command+=(--relation-taxonomy "$taxonomy")
  fi
  printf '%q ' "${command[@]}" > "$output_dir/collapse_command.sh"
  printf '\n' >> "$output_dir/collapse_command.sh"
  echo "[run collapse] period=$period representation=$representation"
  "${command[@]}" | tee "$output_dir/collapse.log"
  collapsed_artifact_is_compatible "$operation" "$source_data" "$taxonomy" "$output_dir" \
    || die "Relation collapse returned success but its provenance check failed: $output_dir"
}

checkpoint_is_compatible() {
  local data_path="$1"
  local checkpoint="$2"
  local metrics_path
  metrics_path="$(dirname "$checkpoint")/metrics.json"
  "$PYTHON_BIN" - "$data_path" "$checkpoint" "$metrics_path" "$MODEL_SEED" "$TRAIN_NUM_NEIGHBORS" <<'PY'
import json
import sys
from pathlib import Path

import torch

from training.tie_taxonomy import sha256_file

data_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
metrics_path = Path(sys.argv[3])
expected_seed = int(sys.argv[4])
expected_fanouts = sys.argv[5]

if not data_path.is_file() or not checkpoint_path.is_file() or not metrics_path.is_file():
    raise SystemExit(1)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
run_config = metrics.get("run_config", {})
model_config = checkpoint.get("model_config", {})
perturbation = checkpoint.get("relation_perturbation", {})
valid = (
    checkpoint.get("model_name") == "rgat"
    and int(model_config.get("num_layers", -1)) == 2
    and run_config.get("model") == "rgat"
    and int(run_config.get("num_layers", -1)) == 2
    and int(run_config.get("seed", -1)) == expected_seed
    and str(run_config.get("num_neighbors")) == expected_fanouts
    and perturbation.get("data_sha256") == sha256_file(data_path)
)
raise SystemExit(0 if valid else 1)
PY
}

probe_is_compatible() {
  local data_path="$1"
  local checkpoint="$2"
  local probe="$3"
  "$PYTHON_BIN" - "$data_path" "$checkpoint" "$probe" <<'PY'
import sys
from pathlib import Path

import torch

from training.tie_taxonomy import sha256_file

data_path, checkpoint_path, probe_path = map(Path, sys.argv[1:])
if not data_path.is_file() or not checkpoint_path.is_file() or not probe_path.is_file():
    raise SystemExit(1)
payload = torch.load(probe_path, map_location="cpu", weights_only=False)
valid = (
    int(payload.get("format_version", 0)) == 1
    and payload.get("source_checkpoint_sha256") == sha256_file(checkpoint_path)
    and payload.get("data_sha256") == sha256_file(data_path)
)
raise SystemExit(0 if valid else 1)
PY
}

report_is_compatible() {
  local data_path="$1"
  local checkpoint="$2"
  local probe="$3"
  local output_dir="$4"
  local filename
  for filename in "${REPORT_FILES[@]}"; do
    [[ -s "$output_dir/$filename" ]] || return 1
  done
  "$PYTHON_BIN" - "$data_path" "$checkpoint" "$probe" "$output_dir/manifest.json" \
    "$GRAPHMASK_NUM_NEIGHBORS" "$GRAPHMASK_TOP_K" <<'PY'
import json
import sys
from pathlib import Path

from training.tie_taxonomy import sha256_file

data_path = Path(sys.argv[1])
checkpoint_path = Path(sys.argv[2])
probe_path = Path(sys.argv[3])
manifest_path = Path(sys.argv[4])
requested_neighbours = sys.argv[5]
top_k = int(sys.argv[6])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
valid = (
    manifest.get("artifact") == "graphmask_report"
    and manifest.get("data_sha256") == sha256_file(data_path)
    and manifest.get("source_checkpoint_sha256") == sha256_file(checkpoint_path)
    and manifest.get("probe_sha256") == sha256_file(probe_path)
    and int(manifest.get("top_k_per_root_per_layer", -1)) == top_k
)
if requested_neighbours in {"full", "-1,-1"}:
    valid = valid and manifest.get("sampling_scope") == "full-neighborhood"
raise SystemExit(0 if valid else 1)
PY
}

write_command() {
  local path="$1"
  shift
  printf '%q ' "$@" > "$path"
  printf '\n' >> "$path"
}

run_period_preparation() {
  local command=(
    "$PYTHON_BIN" scripts/prepare_life_period_induced_artifacts.py
    --source-data "$SOURCE_DATA"
    --output-root "$PERIOD_ARTIFACT_ROOT"
    --life-periods "$LIFE_PERIOD_CONFIG"
    --split-seed "$SPLIT_SEED"
  )
  echo "[run prepare] all historical life-period artifacts"
  "${command[@]}"
}

COMMAND=()

build_model_train_command() {
  local period="$1"
  local representation="$2"
  local data_path taxonomy output_dir
  data_path="$(artifact_path "$period" "$representation")"
  taxonomy="$(tie_taxonomy_for_representation "$period" "$representation")"
  output_dir="$(model_dir "$period" "$representation")"
  COMMAND=(
    "$PYTHON_BIN" run.py train
    --model rgat \
    --data "$data_path" \
    --output-dir "$output_dir" \
    --epochs 50 \
    --train-mode sampled \
    --eval-mode sampled \
    --batch-size "$TRAIN_BATCH_SIZE" \
    --num-layers 2 \
    --num-neighbors "$TRAIN_NUM_NEIGHBORS" \
    --hidden-dim 128 \
    --branch-dim 64 \
    --heads 4 \
    --dropout 0.2 \
    --attention-dropout 0.1 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --early-stop-metric macro_f1 \
    --min-delta 0.001 \
    --patience 6 \
    --num-workers "$TRAIN_NUM_WORKERS" \
    --occupation-feature-levels "$OCCUPATION_FEATURE_LEVELS" \
    --auxiliary-features none \
    --tie-taxonomy "$taxonomy" \
    --seed "$MODEL_SEED" \
    --device "$DEVICE"
  )
}

build_graphmask_train_command() {
  local period="$1"
  local representation="$2"
  local data_path checkpoint output_dir
  data_path="$(artifact_path "$period" "$representation")"
  checkpoint="$(checkpoint_path "$period" "$representation")"
  output_dir="$(graphmask_dir "$period" "$representation")"
  COMMAND=(
    "$PYTHON_BIN" run.py graphmask-train
    --data "$data_path" \
    --checkpoint "$checkpoint" \
    --output-dir "$output_dir" \
    --num-neighbors "$GRAPHMASK_NUM_NEIGHBORS" \
    --batch-size "$GRAPHMASK_BATCH_SIZE" \
    --num-workers "$GRAPHMASK_NUM_WORKERS" \
    --epochs-per-layer "$GRAPHMASK_EPOCHS_PER_LAYER" \
    --seed "$MODEL_SEED" \
    --device "$DEVICE"
  )
}

build_graphmask_report_command() {
  local period="$1"
  local representation="$2"
  local data_path checkpoint probe output_dir
  data_path="$(artifact_path "$period" "$representation")"
  checkpoint="$(checkpoint_path "$period" "$representation")"
  probe="$(probe_path "$period" "$representation")"
  output_dir="$(report_dir "$period" "$representation")"
  COMMAND=(
    "$PYTHON_BIN" run.py graphmask-report
    --data "$data_path" \
    --checkpoint "$checkpoint" \
    --probe "$probe" \
    --output-dir "$output_dir" \
    --split test \
    --num-neighbors "$GRAPHMASK_NUM_NEIGHBORS" \
    --top-k "$GRAPHMASK_TOP_K" \
    --device "$DEVICE"
  )
}

LAST_STATUS=""

run_model_training() {
  local period="$1"
  local representation="$2"
  local data_path checkpoint output_dir
  data_path="$(artifact_path "$period" "$representation")"
  checkpoint="$(checkpoint_path "$period" "$representation")"
  output_dir="$(model_dir "$period" "$representation")"

  if [[ -e "$checkpoint" || -e "$output_dir/metrics.json" ]]; then
    if checkpoint_is_compatible "$data_path" "$checkpoint"; then
      LAST_STATUS="skipped_complete"
      echo "[skip model] period=$period representation=$representation"
      return 0
    fi
    echo "[invalid model] period=$period representation=$representation: $output_dir" >&2
    return 1
  fi

  mkdir -p "$output_dir"
  build_model_train_command "$period" "$representation"
  write_command "$output_dir/train_command.sh" "${COMMAND[@]}"
  echo "[run model] period=$period representation=$representation $(date '+%F %T')"
  if "${COMMAND[@]}" 2>&1 | tee "$output_dir/train.log"; then
    if checkpoint_is_compatible "$data_path" "$checkpoint"; then
      LAST_STATUS="completed"
      return 0
    fi
    echo "[invalid model] Training returned success but checkpoint provenance/configuration is wrong." >&2
    return 1
  fi
  return 1
}

run_graphmask_probe() {
  local period="$1"
  local representation="$2"
  local data_path checkpoint probe output_dir
  data_path="$(artifact_path "$period" "$representation")"
  checkpoint="$(checkpoint_path "$period" "$representation")"
  probe="$(probe_path "$period" "$representation")"
  output_dir="$(graphmask_dir "$period" "$representation")"

  if [[ -e "$probe" ]]; then
    if probe_is_compatible "$data_path" "$checkpoint" "$probe"; then
      LAST_STATUS="skipped_complete"
      echo "[skip probe] period=$period representation=$representation"
      return 0
    fi
    echo "[invalid probe] period=$period representation=$representation: $probe" >&2
    return 1
  fi

  mkdir -p "$output_dir"
  build_graphmask_train_command "$period" "$representation"
  write_command "$output_dir/graphmask_train_command.sh" "${COMMAND[@]}"
  echo "[run probe] period=$period representation=$representation $(date '+%F %T')"
  if "${COMMAND[@]}" 2>&1 | tee "$output_dir/graphmask_train.log"; then
    if probe_is_compatible "$data_path" "$checkpoint" "$probe"; then
      LAST_STATUS="completed"
      return 0
    fi
    echo "[invalid probe] Probe training returned success but source hashes do not match." >&2
    return 1
  fi
  return 1
}

run_graphmask_report() {
  local period="$1"
  local representation="$2"
  local data_path checkpoint probe output_dir
  data_path="$(artifact_path "$period" "$representation")"
  checkpoint="$(checkpoint_path "$period" "$representation")"
  probe="$(probe_path "$period" "$representation")"
  output_dir="$(report_dir "$period" "$representation")"

  if report_is_compatible "$data_path" "$checkpoint" "$probe" "$output_dir"; then
    LAST_STATUS="skipped_complete"
    echo "[skip report] period=$period representation=$representation"
    return 0
  fi

  mkdir -p "$output_dir"
  build_graphmask_report_command "$period" "$representation"
  write_command "$(graphmask_dir "$period" "$representation")/graphmask_report_command.sh" "${COMMAND[@]}"
  echo "[run report] period=$period representation=$representation $(date '+%F %T')"
  if "${COMMAND[@]}" 2>&1 | tee "$(graphmask_dir "$period" "$representation")/graphmask_report.log"; then
    if report_is_compatible "$data_path" "$checkpoint" "$probe" "$output_dir"; then
      LAST_STATUS="completed"
      return 0
    fi
    echo "[invalid report] Report returned success but its manifest is incomplete or incompatible." >&2
    return 1
  fi
  return 1
}

SUMMARY_PATH="$GRAPHMASK_ROOT/period_relation_graphmask_summary.tsv"
FAILURE_PATH="$GRAPHMASK_ROOT/period_relation_graphmask_failures.tsv"
FAILURE_COUNT=0

ensure_status_files() {
  mkdir -p "$GRAPHMASK_ROOT"
  if [[ ! -s "$SUMMARY_PATH" ]]; then
    printf 'timestamp_utc\tperiod\trepresentation\tmodel\tprobe\treport\n' > "$SUMMARY_PATH"
  fi
  if [[ ! -s "$FAILURE_PATH" ]]; then
    printf 'timestamp_utc\tperiod\trepresentation\tstage\tlog_path\n' > "$FAILURE_PATH"
  fi
}

record_summary() {
  local period="$1"
  local representation="$2"
  local model_status="$3"
  local probe_status="$4"
  local report_status="$5"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$period" "$representation" \
    "$model_status" "$probe_status" "$report_status" >> "$SUMMARY_PATH"
}

record_failure() {
  local period="$1"
  local representation="$2"
  local stage="$3"
  local log_path="$4"
  FAILURE_COUNT=$((FAILURE_COUNT + 1))
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$period" "$representation" "$stage" "$log_path" \
    >> "$FAILURE_PATH"
}

run_complete_job() {
  local period="$1"
  local representation="$2"
  local model_status probe_status report_status

  if ! run_model_training "$period" "$representation"; then
    model_status="failed"
    probe_status="not_run"
    report_status="not_run"
    record_failure "$period" "$representation" model "$(model_dir "$period" "$representation")/train.log"
    record_summary "$period" "$representation" "$model_status" "$probe_status" "$report_status"
    echo "[failed model] period=$period representation=$representation; continuing" >&2
    return
  fi
  model_status="$LAST_STATUS"

  if ! run_graphmask_probe "$period" "$representation"; then
    probe_status="failed"
    report_status="not_run"
    record_failure "$period" "$representation" probe "$(graphmask_dir "$period" "$representation")/graphmask_train.log"
    record_summary "$period" "$representation" "$model_status" "$probe_status" "$report_status"
    echo "[failed probe] period=$period representation=$representation; continuing" >&2
    return
  fi
  probe_status="$LAST_STATUS"

  if ! run_graphmask_report "$period" "$representation"; then
    report_status="failed"
    record_failure "$period" "$representation" report "$(graphmask_dir "$period" "$representation")/graphmask_report.log"
    record_summary "$period" "$representation" "$model_status" "$probe_status" "$report_status"
    echo "[failed report] period=$period representation=$representation; continuing" >&2
    return
  fi
  report_status="$LAST_STATUS"
  record_summary "$period" "$representation" "$model_status" "$probe_status" "$report_status"
}

plan_period_preparation() {
  local command=(
    "$PYTHON_BIN" scripts/prepare_life_period_induced_artifacts.py
    --source-data "$SOURCE_DATA"
    --output-root "$PERIOD_ARTIFACT_ROOT"
    --life-periods "$LIFE_PERIOD_CONFIG"
    --split-seed "$SPLIT_SEED"
  )
  echo "[plan prepare] four induced life-period artifacts"
  show_command "${command[@]}"
}

plan_collapses() {
  local period representation source_data output_dir taxonomy command_name
  while IFS= read -r period; do
    [[ -n "$period" ]] || continue
    source_data="$(period_graph_path "$period")"
    for representation in binary multi_group; do
      output_dir="$(representation_dir "$period" "$representation")"
      if [[ "$representation" == "binary" ]]; then
        taxonomy="$TIE_TAXONOMY_PATH"
        command_name="collapse-ties"
      else
        taxonomy="$MULTI_GROUP_TAXONOMY_PATH"
        command_name="collapse-relations"
      fi
      echo "[plan collapse] period=$period representation=$representation"
      if [[ "$representation" == "binary" ]]; then
        show_command "$PYTHON_BIN" run.py "$command_name" --data "$source_data" --tie-taxonomy "$taxonomy" --output-dir "$output_dir"
      else
        show_command "$PYTHON_BIN" run.py "$command_name" --data "$source_data" --relation-taxonomy "$taxonomy" --output-dir "$output_dir"
      fi
    done
  done < <(load_periods)
}

plan_jobs() {
  local period representation
  while IFS= read -r period; do
    [[ -n "$period" ]] || continue
    for representation in "${REPRESENTATIONS[@]}"; do
      echo "[plan job] period=$period representation=$representation model=rgat seed=$MODEL_SEED"
      build_model_train_command "$period" "$representation"
      show_command "${COMMAND[@]}"
      build_graphmask_train_command "$period" "$representation"
      show_command "${COMMAND[@]}"
      build_graphmask_report_command "$period" "$representation"
      show_command "${COMMAND[@]}"
    done
  done < <(load_periods)
}

if [[ "$MODE" == "plan" ]]; then
  echo "Period relation GraphMask matrix: 4 periods × 3 independently trained two-layer R-GATs × seed $MODEL_SEED."
  echo "Training fan-outs=$TRAIN_NUM_NEIGHBORS; GraphMask neighbourhoods=$GRAPHMASK_NUM_NEIGHBORS."
  if selected prepare; then plan_period_preparation; fi
  if selected collapse; then plan_collapses; fi
  if selected train || selected graphmask; then plan_jobs; fi
  exit 0
fi

require_base_inputs
if selected prepare; then
  require_file "$SOURCE_DATA"
  run_period_preparation
fi
if selected collapse; then
  require_period_artifacts
  while IFS= read -r period; do
    [[ -n "$period" ]] || continue
    collapse_artifact binary "$period"
    collapse_artifact multi_group "$period"
  done < <(load_periods)
fi

if selected train || selected graphmask; then
  require_relation_artifacts
  ensure_status_files
  echo "Starting period relation matrix at $(date '+%F %T')"
  echo "device=$DEVICE model_seed=$MODEL_SEED train_fanouts=$TRAIN_NUM_NEIGHBORS graphmask_fanouts=$GRAPHMASK_NUM_NEIGHBORS"
  while IFS= read -r period; do
    [[ -n "$period" ]] || continue
    for representation in "${REPRESENTATIONS[@]}"; do
      if selected train && ! selected graphmask; then
        if run_model_training "$period" "$representation"; then
          record_summary "$period" "$representation" "$LAST_STATUS" "not_requested" "not_requested"
        else
          record_failure "$period" "$representation" model "$(model_dir "$period" "$representation")/train.log"
          record_summary "$period" "$representation" "failed" "not_requested" "not_requested"
          echo "[failed model] period=$period representation=$representation; continuing" >&2
        fi
      elif selected graphmask && ! selected train; then
        if ! checkpoint_is_compatible "$(artifact_path "$period" "$representation")" "$(checkpoint_path "$period" "$representation")"; then
          record_failure "$period" "$representation" model "$(model_dir "$period" "$representation")/train.log"
          record_summary "$period" "$representation" "missing_or_incompatible" "not_run" "not_run"
          echo "[missing/incompatible model] period=$period representation=$representation; continuing" >&2
          continue
        fi
        model_status="skipped_existing"
        if run_graphmask_probe "$period" "$representation"; then
          probe_status="$LAST_STATUS"
        else
          record_failure "$period" "$representation" probe "$(graphmask_dir "$period" "$representation")/graphmask_train.log"
          record_summary "$period" "$representation" "$model_status" "failed" "not_run"
          echo "[failed probe] period=$period representation=$representation; continuing" >&2
          continue
        fi
        if run_graphmask_report "$period" "$representation"; then
          record_summary "$period" "$representation" "$model_status" "$probe_status" "$LAST_STATUS"
        else
          record_failure "$period" "$representation" report "$(graphmask_dir "$period" "$representation")/graphmask_report.log"
          record_summary "$period" "$representation" "$model_status" "$probe_status" "failed"
          echo "[failed report] period=$period representation=$representation; continuing" >&2
        fi
      else
        run_complete_job "$period" "$representation"
      fi
    done
  done < <(load_periods)
  echo "Finished period relation matrix at $(date '+%F %T')"
  echo "Summary: $SUMMARY_PATH"
  if (( FAILURE_COUNT )); then
    echo "$FAILURE_COUNT stage(s) failed; see $FAILURE_PATH" >&2
    exit 1
  fi
fi
