#!/usr/bin/env bash
# DBpedia 2022 occupation-priority graph -> eight life periods -> GPU R-GCN -> GraphMask.
# Usage: bash DBpedia/run_rgcn_graphmask_20y.sh plan|run [all|data|train|graphmask]

set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

MODE="${1:-plan}"
STAGE="${2:-all}"
if [[ "$MODE" != plan && "$MODE" != run ]]; then
  echo "Usage: bash DBpedia/run_rgcn_graphmask_20y.sh plan|run [all|data|train|graphmask]" >&2
  exit 2
fi
if [[ "$STAGE" != all && "$STAGE" != data && "$STAGE" != train && "$STAGE" != graphmask ]]; then
  echo "Unknown stage: $STAGE" >&2
  exit 2
fi

PYTHON_BIN="${DBPEDIA_PYTHON_BIN:-python}"
RAW="${DBPEDIA_RAW_DIR:-external_data/dbpedia/2022.12.01}"
PROCESSED="${DBPEDIA_PROCESSED_DIR:-external_data/dbpedia/processed}"
PERIOD_CONFIG="${DBPEDIA_PERIOD_CONFIG:-config/dbpedia_life_periods_20y_v1.json}"
TIE_TAXONOMY="${DBPEDIA_TIE_TAXONOMY:-config/dbpedia_tie_taxonomy_v1.json}"
FULL_ARTIFACT="${DBPEDIA_FULL_ARTIFACT:-artifacts/dbpedia_2022_priority_v1}"
PERIOD_ROOT="${DBPEDIA_PERIOD_ROOT:-artifacts/dbpedia_2022_priority_periods_20y_v1}"
MODEL_ROOT="${DBPEDIA_MODEL_ROOT:-runs/dbpedia_2022_priority_20y_v1}"
GRAPHMASK_ROOT="${DBPEDIA_GRAPHMASK_ROOT:-runs_graphmask/dbpedia_2022_priority_20y_v1}"
DEVICE="${DBPEDIA_DEVICE:-cuda:0}"
MODEL_SEED="${DBPEDIA_MODEL_SEED:-42}"
SPLIT_SEED="${DBPEDIA_PERIOD_SPLIT_SEED:-20260814}"
TRAIN_EPOCHS="${DBPEDIA_TRAIN_EPOCHS:-50}"
TRAIN_BATCH_SIZE="${DBPEDIA_TRAIN_BATCH_SIZE:-512}"
TRAIN_WORKERS="${DBPEDIA_TRAIN_WORKERS:-4}"
TRAIN_FANOUTS="${DBPEDIA_TRAIN_FANOUTS:-15,10}"
GRAPHMASK_BATCH_SIZE="${DBPEDIA_GRAPHMASK_BATCH_SIZE:-32}"
GRAPHMASK_WORKERS="${DBPEDIA_GRAPHMASK_WORKERS:-0}"
GRAPHMASK_EPOCHS_PER_LAYER="${DBPEDIA_GRAPHMASK_EPOCHS_PER_LAYER:-3}"
GRAPHMASK_BETA="${DBPEDIA_GRAPHMASK_BETA:-0.03}"
GRAPHMASK_MAX_RELATIVE_F1_DIFF="${DBPEDIA_GRAPHMASK_MAX_RELATIVE_F1_DIFF:-0.05}"
GRAPHMASK_TOP_K="${DBPEDIA_GRAPHMASK_TOP_K:-50}"
INPUT_CSV="$PROCESSED/model_input/DBpedia_R_R_extended.csv"

die() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -s "$1" ]] || die "Required file missing or empty: $1"; }
show() { printf '  '; printf '%q ' "$@"; printf '\n'; }
log_run() {
  local log="$1"
  shift
  mkdir -p "$(dirname "$log")"
  echo "[run] $*"
  "$@" 2>&1 | tee "$log"
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] \
  || die "Python executable unavailable: $PYTHON_BIN"
require_file "$PERIOD_CONFIG"
require_file "$TIE_TAXONOMY"

PERIOD_IDS=()
while IFS= read -r period_id; do
  [[ -n "$period_id" ]] && PERIOD_IDS+=("$period_id")
done < <("$PYTHON_BIN" - "$PERIOD_CONFIG" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
for period in config["periods"]:
    print(period["id"])
PY
)
(( ${#PERIOD_IDS[@]} > 0 )) || die "No periods in $PERIOD_CONFIG"
CONTEXTS=(full "${PERIOD_IDS[@]}")

graph_for() {
  local context="$1"
  if [[ "$context" == full ]]; then
    printf '%s/graph_data.pt' "$FULL_ARTIFACT"
  else
    printf '%s/%s/graph_data.pt' "$PERIOD_ROOT" "$context"
  fi
}
model_dir_for() { printf '%s/%s/seed_%s' "$MODEL_ROOT" "$1" "$MODEL_SEED"; }
mask_dir_for() { printf '%s/%s/seed_%s' "$GRAPHMASK_ROOT" "$1" "$MODEL_SEED"; }

train_command() {
  local context="$1" data="$2" output="$3"
  COMMAND=(
    "$PYTHON_BIN" run.py train
    --model rgcn --data "$data" --output-dir "$output"
    --epochs "$TRAIN_EPOCHS" --train-mode sampled --eval-mode sampled
    --batch-size "$TRAIN_BATCH_SIZE" --num-layers 2 --num-neighbors "$TRAIN_FANOUTS"
    --hidden-dim 128 --branch-dim 64 --num-bases 30 --rgcn-backend fast
    --dropout 0.2 --lr 0.001 --weight-decay 0.0001
    --early-stop-metric macro_f1 --min-delta 0.001 --patience 6
    --num-workers "$TRAIN_WORKERS" --occupation-feature-levels 1
    --auxiliary-features temporal --tie-taxonomy "$TIE_TAXONOMY"
    --seed "$MODEL_SEED" --device "$DEVICE"
  )
}
mask_train_command() {
  local data="$1" checkpoint="$2" output="$3"
  COMMAND=(
    "$PYTHON_BIN" run.py graphmask-train
    --data "$data" --checkpoint "$checkpoint" --output-dir "$output"
    --num-neighbors auto --batch-size "$GRAPHMASK_BATCH_SIZE"
    --num-workers "$GRAPHMASK_WORKERS"
    --epochs-per-layer "$GRAPHMASK_EPOCHS_PER_LAYER"
    --beta "$GRAPHMASK_BETA"
    --max-relative-macro-f1-diff "$GRAPHMASK_MAX_RELATIVE_F1_DIFF"
    --seed "$MODEL_SEED" --device "$DEVICE"
  )
}
mask_report_command() {
  local data="$1" checkpoint="$2" probe="$3" output="$4"
  COMMAND=(
    "$PYTHON_BIN" run.py graphmask-report
    --data "$data" --checkpoint "$checkpoint" --probe "$probe"
    --output-dir "$output" --split test --num-neighbors auto
    --top-k "$GRAPHMASK_TOP_K" --device "$DEVICE"
  )
}

model_complete() {
  local dir="$1"
  [[ -s "$dir/best_model.pt" && -s "$dir/metrics.json" && -s "$dir/test_predictions.csv" ]]
}
probe_complete() {
  local dir="$1"
  [[ -s "$dir/graphmask_probe.pt" && -s "$dir/manifest.json" && -s "$dir/validation.json" ]]
}
report_complete() {
  local dir="$1" name
  for name in test_metrics.json relations_directed.csv relations_base.csv root_top_edges.csv.gz manifest.json; do
    [[ -s "$dir/$name" ]] || return 1
  done
  return 0
}

gpu_preflight() {
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} device=$DEVICE"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
  fi
  "$PYTHON_BIN" - "$DEVICE" <<'PY'
import sys
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

device = torch.device(sys.argv[1])
if device.type != "cuda" or not torch.cuda.is_available():
    raise SystemExit("CUDA is required for this GPU run; check the Python environment and GPU visibility")
torch.empty(1, device=device)
print("torch", torch.__version__, "torch CUDA", torch.version.cuda)
print("selected GPU", torch.cuda.get_device_name(device))
sample = Data(edge_index=torch.tensor([[0, 1], [1, 0]]), num_nodes=2)
loader = NeighborLoader(sample, num_neighbors=[1, 1], input_nodes=torch.tensor([0]), batch_size=1, num_workers=0)
next(iter(loader))
print("PyG NeighborLoader OK")
PY
}

plan_data() {
  echo "[data] raw: $RAW/"
  show "$PYTHON_BIN" DBpedia/run.py --raw-dir "$RAW" --output-dir "$PROCESSED"
  echo "[data] source: $PROCESSED/04_graph/"
  show "$PYTHON_BIN" DBpedia/export_q_r_q_extended.py --processed-dir "$PROCESSED"
  show "$PYTHON_BIN" run.py prepare --input "$INPUT_CSV" --output-dir "$FULL_ARTIFACT" --target-level 1 --min-class-count 20 --seed 42
  show "$PYTHON_BIN" scripts/prepare_life_period_induced_artifacts.py --source-data "$FULL_ARTIFACT/graph_data.pt" --output-root "$PERIOD_ROOT" --life-periods "$PERIOD_CONFIG" --split-seed "$SPLIT_SEED"
}

run_data() {
  if [[ -s "$PROCESSED/04_graph/graph_nodes.tsv.gz" && -s "$PROCESSED/04_graph/graph_edges.tsv.gz" && -s "$PROCESSED/04_graph/candidate_evidence.tsv.gz" && -s "$PROCESSED/04_graph/summary.json" ]]; then
    echo "[skip] four-stage extraction exists: $PROCESSED/04_graph/"
  else
    mkdir -p "$PROCESSED"
    log_run "$PROCESSED/extraction.log" \
      "$PYTHON_BIN" DBpedia/run.py --raw-dir "$RAW" --output-dir "$PROCESSED"
  fi
  require_file "$PROCESSED/04_graph/graph_nodes.tsv.gz"
  require_file "$PROCESSED/04_graph/graph_edges.tsv.gz"
  require_file "$PROCESSED/04_graph/candidate_evidence.tsv.gz"
  if [[ -s "$INPUT_CSV" && -s "$PROCESSED/model_input/summary.json" && -s "$PROCESSED/model_input/occupation_assignments.tsv.gz" ]]; then
    echo "[skip] DBpedia compatible CSV exists: $INPUT_CSV"
  else
    log_run "$PROCESSED/model_input/export.log" \
      "$PYTHON_BIN" DBpedia/export_q_r_q_extended.py --processed-dir "$PROCESSED"
  fi
  if [[ -s "$FULL_ARTIFACT/graph_data.pt" && -s "$FULL_ARTIFACT/nodes.csv" && -s "$FULL_ARTIFACT/edges.csv" && -s "$FULL_ARTIFACT/split_summary.json" ]]; then
    echo "[skip] full graph artifact exists: $FULL_ARTIFACT/graph_data.pt"
  else
    if [[ -e "$FULL_ARTIFACT" ]]; then
      die "Incomplete full artifact at $FULL_ARTIFACT; inspect it before rerunning"
    fi
    log_run "$MODEL_ROOT/data_prepare.log" \
      "$PYTHON_BIN" run.py prepare --input "$INPUT_CSV" --output-dir "$FULL_ARTIFACT" \
      --target-level 1 --min-class-count 20 --seed 42
  fi
  log_run "$MODEL_ROOT/period_prepare.log" \
    "$PYTHON_BIN" scripts/prepare_life_period_induced_artifacts.py \
    --source-data "$FULL_ARTIFACT/graph_data.pt" --output-root "$PERIOD_ROOT" \
    --life-periods "$PERIOD_CONFIG" --split-seed "$SPLIT_SEED"
}

run_models() {
  local context data output
  for context in "${CONTEXTS[@]}"; do
    data="$(graph_for "$context")"
    require_file "$data"
    output="$(model_dir_for "$context")"
    if model_complete "$output"; then
      echo "[skip] R-GCN complete: $context"
      continue
    fi
    mkdir -p "$output"
    train_command "$context" "$data" "$output"
    log_run "$output/train.log" "${COMMAND[@]}"
    model_complete "$output" || die "R-GCN returned without complete outputs: $context"
  done
}

run_graphmask() {
  local context data model_dir checkpoint output report
  for context in "${CONTEXTS[@]}"; do
    data="$(graph_for "$context")"
    model_dir="$(model_dir_for "$context")"
    checkpoint="$model_dir/best_model.pt"
    output="$(mask_dir_for "$context")"
    report="$output/test_report"
    require_file "$data"
    model_complete "$model_dir" || die "R-GCN checkpoint/metrics/test predictions missing for $context"
    mkdir -p "$output"
    if probe_complete "$output"; then
      echo "[skip] GraphMask probe complete: $context"
    else
      mask_train_command "$data" "$checkpoint" "$output"
      log_run "$output/graphmask_train.log" "${COMMAND[@]}"
      probe_complete "$output" || die "GraphMask probe absent after training: $context"
    fi
    if report_complete "$report"; then
      echo "[skip] GraphMask test report complete: $context"
    else
      mask_report_command "$data" "$checkpoint" "$output/graphmask_probe.pt" "$report"
      log_run "$output/graphmask_report.log" "${COMMAND[@]}"
      report_complete "$report" || die "GraphMask test report incomplete: $context"
    fi
  done
  "$PYTHON_BIN" DBpedia/summarize_graphmask_matrix.py \
    --full-artifact "$FULL_ARTIFACT" --period-root "$PERIOD_ROOT" \
    --graphmask-root "$GRAPHMASK_ROOT" --seed "$MODEL_SEED" \
    --period-config "$PERIOD_CONFIG"
}

echo "DBpedia experiment: mode=$MODE stage=$STAGE contexts=${#CONTEXTS[@]} device=$DEVICE"
if [[ "$MODE" == plan ]]; then
  if [[ "$STAGE" == all || "$STAGE" == data ]]; then plan_data; fi
  if [[ "$STAGE" == all || "$STAGE" == train || "$STAGE" == graphmask ]]; then
    for context in "${CONTEXTS[@]}"; do
      data="$(graph_for "$context")"
      model_dir="$(model_dir_for "$context")"
      output="$(mask_dir_for "$context")"
      if [[ "$STAGE" == all || "$STAGE" == train ]]; then
        train_command "$context" "$data" "$model_dir"
        echo "[train] $context $(model_complete "$model_dir" && echo SKIP || echo RUN)"
        show "${COMMAND[@]}"
      fi
      if [[ "$STAGE" == all || "$STAGE" == graphmask ]]; then
        mask_train_command "$data" "$model_dir/best_model.pt" "$output"
        echo "[graphmask] $context $(probe_complete "$output" && echo SKIP || echo RUN)"
        show "${COMMAND[@]}"
        mask_report_command "$data" "$model_dir/best_model.pt" "$output/graphmask_probe.pt" "$output/test_report"
        echo "[report] $context $(report_complete "$output/test_report" && echo SKIP || echo RUN)"
        show "${COMMAND[@]}"
      fi
    done
  fi
  exit 0
fi

trap 'echo "[failed] line $LINENO, stage=$STAGE; inspect the job log if a job started" >&2' ERR
if [[ "$STAGE" == all || "$STAGE" == train || "$STAGE" == graphmask ]]; then
  gpu_preflight
fi
if [[ "$STAGE" == all || "$STAGE" == data ]]; then run_data; fi
if [[ "$STAGE" == all || "$STAGE" == train ]]; then run_models; fi
if [[ "$STAGE" == all || "$STAGE" == graphmask ]]; then run_graphmask; fi
echo "[done] DBpedia stage=$STAGE"
