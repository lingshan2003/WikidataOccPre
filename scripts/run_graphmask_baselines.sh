#!/usr/bin/env bash
# Train and report one GraphMask probe (seed 42) for every L1/L2/L3 baseline.
#
# The matrix is deliberately sequential so that only one GPU-heavy process is
# active at a time. It is restart-safe: a valid graphmask_probe.pt skips probe
# training, and a complete five-file test_report skips reporting. Failures are
# logged and recorded while the remaining independent jobs continue.

set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
PYTHON_BIN="${RGCN_PYTHON_BIN:-python}"
OUTPUT_ROOT="${RGCN_GRAPHMASK_OUTPUT_ROOT:-runs_graphmask}"
DEVICE="${RGCN_GRAPHMASK_DEVICE:-cuda:0}"
NUM_NEIGHBORS="${RGCN_GRAPHMASK_NUM_NEIGHBORS:-full}"
BATCH_SIZE="${RGCN_GRAPHMASK_BATCH_SIZE:-32}"
NUM_WORKERS="${RGCN_GRAPHMASK_NUM_WORKERS:-4}"
EPOCHS_PER_LAYER="${RGCN_GRAPHMASK_EPOCHS_PER_LAYER:-3}"
TOP_K="${RGCN_GRAPHMASK_TOP_K:-50}"
PROBE_SEED="${RGCN_GRAPHMASK_SEED:-42}"

LEVELS=(1 2 3)
MODELS=(rgcn rgat compgcn)
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
  bash scripts/run_graphmask_baselines.sh plan
  bash scripts/run_graphmask_baselines.sh run

The script uses seed 42 only and processes these jobs sequentially:
  level{1,2,3} x {rgcn,rgat,compgcn}_baseline

Restart behavior:
  - graphmask_probe.pt exists: skip GraphMask training.
  - all five test_report outputs exist: skip GraphMask reporting.
  - a failed job is logged, then the script continues with the next model.

Optional environment overrides:
  RGCN_PYTHON_BIN=.venv/bin/python
  RGCN_GRAPHMASK_OUTPUT_ROOT=runs_graphmask
  RGCN_GRAPHMASK_DEVICE=cuda:0
  RGCN_GRAPHMASK_NUM_NEIGHBORS=full
  RGCN_GRAPHMASK_BATCH_SIZE=32
  RGCN_GRAPHMASK_NUM_WORKERS=4
  RGCN_GRAPHMASK_EPOCHS_PER_LAYER=3
  RGCN_GRAPHMASK_TOP_K=50
  RGCN_GRAPHMASK_SEED=42

For a less expensive run, set RGCN_GRAPHMASK_NUM_NEIGHBORS=auto. When using
full neighborhoods and CUDA runs out of memory, reduce
RGCN_GRAPHMASK_BATCH_SIZE before restarting; completed jobs remain skipped.
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then
  usage
  exit 2
fi

show_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

report_is_complete() {
  local report_dir="$1"
  local filename
  for filename in "${REPORT_FILES[@]}"; do
    if [[ ! -s "$report_dir/$filename" ]]; then
      return 1
    fi
  done
  return 0
}

data_path_for_level() {
  local level="$1"
  printf 'artifacts/level%s_hierarchy/graph_data.pt' "$level"
}

checkpoint_path_for_job() {
  local level="$1"
  local model="$2"
  printf 'runs_report/level%s/%s_baseline/seed_%s/best_model.pt' \
    "$level" "$model" "$PROBE_SEED"
}

output_dir_for_job() {
  local level="$1"
  local model="$2"
  printf '%s/level%s_%s/seed_%s' "$OUTPUT_ROOT" "$level" "$model" "$PROBE_SEED"
}

require_inputs() {
  local missing=0
  local level model data_path checkpoint_path
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable is unavailable: $PYTHON_BIN" >&2
    missing=1
  fi
  for level in "${LEVELS[@]}"; do
    data_path="$(data_path_for_level "$level")"
    if [[ ! -f "$data_path" ]]; then
      echo "Missing graph artifact: $data_path" >&2
      missing=1
    fi
    for model in "${MODELS[@]}"; do
      checkpoint_path="$(checkpoint_path_for_job "$level" "$model")"
      if [[ ! -f "$checkpoint_path" ]]; then
        echo "Missing trained baseline checkpoint: $checkpoint_path" >&2
        missing=1
      fi
    done
  done
  if (( missing )); then
    echo "Input preflight failed; no GraphMask job was started." >&2
    exit 1
  fi
}

plan_job() {
  local level="$1"
  local model="$2"
  local data_path checkpoint_path output_dir report_dir probe_path
  data_path="$(data_path_for_level "$level")"
  checkpoint_path="$(checkpoint_path_for_job "$level" "$model")"
  output_dir="$(output_dir_for_job "$level" "$model")"
  report_dir="$output_dir/test_report"
  probe_path="$output_dir/graphmask_probe.pt"

  local train_command=(
    "$PYTHON_BIN" run.py graphmask-train
    --data "$data_path"
    --checkpoint "$checkpoint_path"
    --output-dir "$output_dir"
    --num-neighbors "$NUM_NEIGHBORS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --epochs-per-layer "$EPOCHS_PER_LAYER"
    --seed "$PROBE_SEED"
    --device "$DEVICE"
  )
  local report_command=(
    "$PYTHON_BIN" run.py graphmask-report
    --data "$data_path"
    --checkpoint "$checkpoint_path"
    --probe "$probe_path"
    --output-dir "$report_dir"
    --split test
    --num-neighbors "$NUM_NEIGHBORS"
    --top-k "$TOP_K"
    --device "$DEVICE"
  )

  printf '[plan] level=%s model=%-8s train=%s report=%s\n' \
    "$level" "$model" \
    "$([[ -s "$probe_path" ]] && printf skip || printf run)" \
    "$(report_is_complete "$report_dir" && printf skip || printf run)"
  if [[ ! -s "$probe_path" ]]; then
    show_command "${train_command[@]}"
  fi
  if ! report_is_complete "$report_dir"; then
    show_command "${report_command[@]}"
  fi
}

SUMMARY_PATH="$OUTPUT_ROOT/graphmask_baseline_summary.tsv"
FAILURE_PATH="$OUTPUT_ROOT/graphmask_baseline_failures.tsv"
FAILED_JOBS=0

record_summary() {
  local level="$1"
  local model="$2"
  local train_status="$3"
  local report_status="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$level" "$model" "$train_status" "$report_status" \
    >> "$SUMMARY_PATH"
}

record_failure() {
  local level="$1"
  local model="$2"
  local stage="$3"
  local log_path="$4"
  FAILED_JOBS=$((FAILED_JOBS + 1))
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$level" "$model" "$stage" "$log_path" \
    >> "$FAILURE_PATH"
}

run_job() {
  local level="$1"
  local model="$2"
  local data_path checkpoint_path output_dir report_dir probe_path
  local train_status report_status
  data_path="$(data_path_for_level "$level")"
  checkpoint_path="$(checkpoint_path_for_job "$level" "$model")"
  output_dir="$(output_dir_for_job "$level" "$model")"
  report_dir="$output_dir/test_report"
  probe_path="$output_dir/graphmask_probe.pt"
  mkdir -p "$output_dir" "$report_dir"

  local train_command=(
    "$PYTHON_BIN" run.py graphmask-train
    --data "$data_path"
    --checkpoint "$checkpoint_path"
    --output-dir "$output_dir"
    --num-neighbors "$NUM_NEIGHBORS"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --epochs-per-layer "$EPOCHS_PER_LAYER"
    --seed "$PROBE_SEED"
    --device "$DEVICE"
  )
  local report_command=(
    "$PYTHON_BIN" run.py graphmask-report
    --data "$data_path"
    --checkpoint "$checkpoint_path"
    --probe "$probe_path"
    --output-dir "$report_dir"
    --split test
    --num-neighbors "$NUM_NEIGHBORS"
    --top-k "$TOP_K"
    --device "$DEVICE"
  )

  if [[ -s "$probe_path" ]]; then
    train_status="skipped_complete"
    echo "[skip train] level=$level model=$model probe exists: $probe_path"
  else
    printf '%q ' "${train_command[@]}" > "$output_dir/graphmask_train_command.sh"
    printf '\n' >> "$output_dir/graphmask_train_command.sh"
    echo "[run train] level=$level model=$model $(date '+%F %T')"
    if "${train_command[@]}" 2>&1 | tee "$output_dir/graphmask_train.log"; then
      if [[ ! -s "$probe_path" ]]; then
        echo "Training returned success but did not create $probe_path" \
          | tee -a "$output_dir/graphmask_train.log" >&2
        train_status="failed_missing_probe"
        report_status="not_run"
        record_failure "$level" "$model" train "$output_dir/graphmask_train.log"
        record_summary "$level" "$model" "$train_status" "$report_status"
        return
      fi
      train_status="completed"
    else
      train_status="failed"
      report_status="not_run"
      record_failure "$level" "$model" train "$output_dir/graphmask_train.log"
      record_summary "$level" "$model" "$train_status" "$report_status"
      echo "[failed train] level=$level model=$model; continuing with the next job" >&2
      return
    fi
  fi

  if report_is_complete "$report_dir"; then
    report_status="skipped_complete"
    echo "[skip report] level=$level model=$model report is complete: $report_dir"
  else
    printf '%q ' "${report_command[@]}" > "$output_dir/graphmask_report_command.sh"
    printf '\n' >> "$output_dir/graphmask_report_command.sh"
    echo "[run report] level=$level model=$model $(date '+%F %T')"
    if "${report_command[@]}" 2>&1 | tee "$output_dir/graphmask_report.log"; then
      if report_is_complete "$report_dir"; then
        report_status="completed"
      else
        report_status="failed_incomplete_outputs"
        record_failure "$level" "$model" report "$output_dir/graphmask_report.log"
      fi
    else
      report_status="failed"
      record_failure "$level" "$model" report "$output_dir/graphmask_report.log"
      echo "[failed report] level=$level model=$model; continuing with the next job" >&2
    fi
  fi
  record_summary "$level" "$model" "$train_status" "$report_status"
}

if [[ "$MODE" == "plan" ]]; then
  echo "GraphMask baseline matrix: 3 levels x 3 models x one seed ($PROBE_SEED)."
  echo "Completed probes/reports are skipped automatically."
  for level in "${LEVELS[@]}"; do
    for model in "${MODELS[@]}"; do
      plan_job "$level" "$model"
    done
  done
  exit 0
fi

require_inputs
mkdir -p "$OUTPUT_ROOT"
printf 'timestamp_utc\tlevel\tmodel\ttrain_status\treport_status\n' > "$SUMMARY_PATH"
printf 'timestamp_utc\tlevel\tmodel\tstage\tlog_path\n' > "$FAILURE_PATH"

echo "Starting GraphMask baseline matrix at $(date '+%F %T')"
echo "Python=$PYTHON_BIN device=$DEVICE neighbors=$NUM_NEIGHBORS batch=$BATCH_SIZE seed=$PROBE_SEED"
for level in "${LEVELS[@]}"; do
  for model in "${MODELS[@]}"; do
    run_job "$level" "$model"
  done
done

echo "GraphMask matrix finished at $(date '+%F %T')"
echo "Summary: $SUMMARY_PATH"
if (( FAILED_JOBS )); then
  echo "$FAILED_JOBS stage(s) failed. See $FAILURE_PATH and the per-job logs." >&2
  exit 1
fi
echo "All requested GraphMask probes and reports are complete."
