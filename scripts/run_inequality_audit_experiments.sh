#!/usr/bin/env bash
# Level-1 relational-information privilege amplification audit.
#
# The matrix isolates message passing with a shared-encoder MLP and adds a
# typed-degree-preserving inherited-tie rewiring control. Every condition is
# trained inside a separate output root and every stage is restart safe.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-plan}"
STAGE="${2:-all}"
PYTHON_BIN="${RGCN_PYTHON_BIN:-python}"
DATA_PATH="${RGCN_INEQUALITY_DATA:-artifacts/level1_hierarchy/graph_data.pt}"
OUTPUT_ROOT="${RGCN_INEQUALITY_OUTPUT_ROOT:-runs_report/level1/inequality_audit}"
TAXONOMY_PATH="${RGCN_INEQUALITY_TAXONOMY:-config/tie_taxonomy_ascribed_family_v1.json}"
DEVICE="${RGCN_INEQUALITY_DEVICE:-cuda:0}"
NUM_WORKERS="${RGCN_INEQUALITY_NUM_WORKERS:-4}"
PREDICT_WORKERS="${RGCN_INEQUALITY_PREDICT_WORKERS:-0}"
BOOTSTRAP_DRAWS="${RGCN_INEQUALITY_BOOTSTRAP_DRAWS:-500}"
read -r -a SEEDS <<< "${RGCN_INEQUALITY_SEEDS:-42 43 44}"
CURRENT_DATA_SHA=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_inequality_audit_experiments.sh plan [all|train|predict|audit]
  bash scripts/run_inequality_audit_experiments.sh run  [all|train|predict|audit]

Typical server launch:
  RGCN_PYTHON_BIN=.venv/bin/python \
  nohup bash scripts/run_inequality_audit_experiments.sh run all \
    > runs_report/level1/inequality_audit.nohup.log 2>&1 &

Important environment overrides:
  RGCN_INEQUALITY_DATA=artifacts/level1_hierarchy/graph_data.pt
  RGCN_INEQUALITY_OUTPUT_ROOT=runs_report/level1/inequality_audit
  RGCN_INEQUALITY_DEVICE=cuda:0
  RGCN_INEQUALITY_SEEDS="42 43 44"
  RGCN_INEQUALITY_BOOTSTRAP_DRAWS=500

Training conditions:
  mlp_baseline                 shared node encoder, no message passing
  rgcn_full                    canonical graph
  rgcn_without_inherited       retrained edge ablation
  rgcn_random_matched_inherited exact edge-count random control
  rgcn_rewired_inherited       same-relation degree-preserving endpoint swaps

Prediction also evaluates rgcn_full_inference_without_inherited using the
frozen full checkpoint, so it requires no additional training.
EOF
}

if [[ "$MODE" != "plan" && "$MODE" != "run" ]]; then
  usage
  exit 2
fi
if [[ ! " all train predict audit " == *" $STAGE "* ]]; then
  usage
  exit 2
fi

selected() {
  [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]
}

show_command() {
  printf ' '
  printf '%q ' "$@"
  printf '\n'
}

fallback_checkpoint() {
  local condition="$1"
  local seed="$2"
  printf '%s/checkpoints/%s/seed_%s/best_model.pt' "$OUTPUT_ROOT" "$condition" "$seed"
}

resolved_checkpoint() {
  local condition="$1"
  local seed="$2"
  fallback_checkpoint "$condition" "$seed"
}

common_train_args=(
  --data "$DATA_PATH"
  --epochs 50
  --batch-size 512
  --num-layers 2
  --hidden-dim 128
  --branch-dim 64
  --early-stop-metric macro_f1
  --min-delta 0.001
  --patience 6
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --occupation-feature-levels 1,2,3
  --auxiliary-features country,temporal
  --tie-taxonomy "$TAXONOMY_PATH"
)

train_condition() {
  local condition="$1"
  local model="$2"
  shift 2
  for seed in "${SEEDS[@]}"; do
    local fallback output_dir
    fallback="$(fallback_checkpoint "$condition" "$seed")"
    output_dir="$(dirname "$fallback")"
    local command=(
      "$PYTHON_BIN" run.py train
      --model "$model"
      --output-dir "$output_dir"
      "${common_train_args[@]}"
      --seed "$seed"
      "$@"
    )
    if [[ "$MODE" == "plan" ]]; then
      printf '%-42s seed=%s' "$condition" "$seed"
      show_command "${command[@]}"
      continue
    fi
    if [[ -f "$fallback" && -f "$output_dir/metrics.json" ]]; then
      echo "[skip] $condition seed=$seed already complete"
      continue
    fi
    mkdir -p "$output_dir"
    printf '%q ' "${command[@]}" > "$output_dir/command.sh"
    printf '\n' >> "$output_dir/command.sh"
    echo "[train] $condition seed=$seed"
    "${command[@]}"
  done
}

run_training_matrix() {
  train_condition mlp_baseline mlp --num-neighbors 0,0
  train_condition rgcn_full rgcn --rgcn-backend fast --num-neighbors 15,10
  train_condition rgcn_without_inherited rgcn --rgcn-backend fast --num-neighbors 15,10 \
    --drop-tie-groups inherited
  train_condition rgcn_random_matched_inherited rgcn --rgcn-backend fast --num-neighbors 15,10 \
    --match-random-drop-to-tie-groups inherited
  train_condition rgcn_rewired_inherited rgcn --rgcn-backend fast --num-neighbors 15,10 \
    --degree-preserving-rewire-tie-groups inherited --rewire-swaps-per-edge 5
}

predict_condition() {
  local condition="$1"
  local checkpoint_condition="$2"
  shift 2
  for seed in "${SEEDS[@]}"; do
    local checkpoint output_dir
    checkpoint="$(resolved_checkpoint "$checkpoint_condition" "$seed")"
    output_dir="$OUTPUT_ROOT/predictions/$condition/seed_$seed"
    local command=(
      "$PYTHON_BIN" run.py inequality-predict
      --data "$DATA_PATH"
      --checkpoint "$checkpoint"
      --output-dir "$output_dir"
      --condition "$condition"
      --seed "$seed"
      --split test
      --num-neighbors auto
      --batch-size 512
      --num-workers "$PREDICT_WORKERS"
      --device "$DEVICE"
      --tie-taxonomy "$TAXONOMY_PATH"
      "$@"
    )
    if [[ "$MODE" == "plan" ]]; then
      printf '%-42s seed=%s' "$condition" "$seed"
      show_command "${command[@]}"
      continue
    fi
    if [[ ! -f "$checkpoint" ]]; then
      echo "Missing checkpoint for $condition seed=$seed: $checkpoint" >&2
      echo "Run the train stage first, or point the baseline roots at completed runs." >&2
      exit 1
    fi
    if prediction_is_current "$output_dir" "$checkpoint" "$condition" "$seed"; then
      echo "[skip] prediction $condition seed=$seed already complete"
      continue
    fi
    mkdir -p "$output_dir"
    printf '%q ' "${command[@]}" > "$output_dir/command.sh"
    printf '\n' >> "$output_dir/command.sh"
    echo "[predict] $condition seed=$seed"
    "${command[@]}"
  done
}

prediction_is_current() {
  local output_dir="$1"
  local checkpoint="$2"
  local condition="$3"
  local seed="$4"
  if [[ ! -f "$output_dir/manifest.json" || ! -f "$output_dir/probabilities.npz" ]]; then
    return 1
  fi
  "$PYTHON_BIN" - "$output_dir/manifest.json" "$checkpoint" "$CURRENT_DATA_SHA" "$condition" "$seed" <<'PY'
import hashlib
import json
import sys

manifest_path, checkpoint_path, data_sha, condition, seed = sys.argv[1:]
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
digest = hashlib.sha256()
with open(checkpoint_path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
valid = (
    manifest.get("condition") == condition
    and int(manifest.get("seed", -1)) == int(seed)
    and manifest.get("data_sha256") == data_sha
    and manifest.get("checkpoint_sha256") == digest.hexdigest()
)
raise SystemExit(0 if valid else 1)
PY
}

run_prediction_matrix() {
  predict_condition mlp_baseline mlp_baseline
  predict_condition rgcn_full rgcn_full
  predict_condition rgcn_without_inherited rgcn_without_inherited
  predict_condition rgcn_random_matched_inherited rgcn_random_matched_inherited
  predict_condition rgcn_rewired_inherited rgcn_rewired_inherited
  predict_condition rgcn_full_inference_without_inherited rgcn_full \
    --inference-drop-tie-groups inherited
}

run_audit() {
  local command=(
    "$PYTHON_BIN" run.py inequality-audit
    --data "$DATA_PATH"
    --predictions-root "$OUTPUT_ROOT/predictions"
    --output-dir "$OUTPUT_ROOT/summary"
    --tie-taxonomy "$TAXONOMY_PATH"
    --bootstrap-draws "$BOOTSTRAP_DRAWS"
    --bootstrap-seed 20260824
  )
  if [[ "$MODE" == "plan" ]]; then
    printf '%-42s' inequality_audit_summary
    show_command "${command[@]}"
  else
    echo "[audit] inequality summary"
    "${command[@]}"
  fi
}

if [[ "$MODE" == "run" ]]; then
  if [[ ! -f "$DATA_PATH" ]]; then
    echo "Missing Level-1 graph artifact: $DATA_PATH" >&2
    exit 1
  fi
  if [[ ! -f "$TAXONOMY_PATH" ]]; then
    echo "Missing tie taxonomy: $TAXONOMY_PATH" >&2
    exit 1
  fi
  "$PYTHON_BIN" - "$DATA_PATH" <<'PY'
import sys
import torch

path = sys.argv[1]
bundle = torch.load(path, map_location="cpu", weights_only=False)
data, metadata = bundle["data"], bundle["metadata"]
if metadata.get("target_column") != "occupation_level1":
    raise SystemExit(f"Expected a Level-1 target artifact, got {metadata.get('target_column')!r}")
required = (
    "occupation_level1", "occupation_level2", "occupation_level3",
    "country", "temporal", "edge_index", "edge_type",
    "train_mask", "val_mask", "test_mask",
)
missing = [name for name in required if not hasattr(data, name)]
if missing:
    raise SystemExit(f"Artifact predates the required feature protocol; missing: {missing}")
print(
    f"[preflight] nodes={data.num_nodes} edges={data.edge_index.size(1)} "
    f"train/val/test={int(data.train_mask.sum())}/{int(data.val_mask.sum())}/{int(data.test_mask.sum())}"
)
PY
  CURRENT_DATA_SHA="$("$PYTHON_BIN" - "$DATA_PATH" <<'PY'
import hashlib
import sys

digest = hashlib.sha256()
with open(sys.argv[1], "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
fi

if selected train; then
  run_training_matrix
fi
if selected predict; then
  run_prediction_matrix
fi
if selected audit; then
  run_audit
fi

if [[ "$MODE" == "plan" ]]; then
  echo "Plan complete: 5 training conditions and 6 frozen-checkpoint prediction conditions per seed."
else
  echo "Inequality audit stage '$STAGE' complete. Results: $OUTPUT_ROOT/summary"
fi
