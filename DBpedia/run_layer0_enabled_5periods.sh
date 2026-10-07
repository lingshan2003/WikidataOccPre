#!/usr/bin/env bash
# Supplementary multi-group GraphMask selection; reuse the five frozen RGCNs.
# Usage: bash DBpedia/run_layer0_enabled_5periods.sh [plan|run|summarize|package]
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PYTHON_BIN="${DBPEDIA_PYTHON_BIN:-python}"
CONFIG="config/dbpedia_multi_group_layer0_enabled_5periods_v1.json"
PERIODS="through_1500,1501_1900,1941_1960,1981_2000,since_2001"
MODE="${1:-plan}"
[[ $# -le 1 ]] || { echo "Usage: $0 [plan|run|summarize|package]" >&2; exit 2; }
# This is a selection-only comparison: numerical overrides from other jobs
# must not silently change the original training settings.
for DBPEDIA_SUPPLEMENT_OPTION in \
  SEED TRAIN_BATCH_SIZE TRAIN_WORKERS TRAIN_EPOCHS TRAIN_FANOUTS \
  GRAPHMASK_BATCH_SIZE GRAPHMASK_WORKERS GRAPHMASK_EPOCHS_PER_LAYER \
  GRAPHMASK_BETA GRAPHMASK_MAX_RELATIVE_F1_DIFF; do
  unset "DBPEDIA_GROUP_${DBPEDIA_SUPPLEMENT_OPTION}"
done
case "$MODE" in
  plan|run|summarize)
    PIPELINE_MODE="$MODE"
    STAGE="graphmask"
    if [[ "$MODE" == "summarize" ]]; then PIPELINE_MODE="run"; STAGE="summarize"; fi
    # Pin this supplement's scope even if a previous run exported wider selections.
    DBPEDIA_GROUP_INCLUDE_FULL=0 \
    DBPEDIA_GROUP_GRAPHMASK_ROOT=runs_graphmask/dbpedia_multi_group_layer0_enabled_5periods_v1 \
    "$PYTHON_BIN" DBpedia/grouped_pipeline.py \
      "$PIPELINE_MODE" "$STAGE" --config "$CONFIG" \
      --periods "$PERIODS" --representations multi_group
    if [[ "$MODE" != "plan" ]]; then
      "$PYTHON_BIN" DBpedia/compare_layer0_checkpoint_selection.py --config "$CONFIG"
    fi
    ;;
  package)
    # Package statistics and provenance, without graph/model weights or large CSVs.
    tar -czf dbpedia_layer0_enabled_5periods_visualization.tar.gz \
      --exclude='*.pt' --exclude='*.log' --exclude='*.lock' \
      --exclude='root_top_edges.csv.gz' --exclude='test_predictions.csv' \
      --exclude='nodes.csv' --exclude='edges.csv' \
      runs_graphmask/dbpedia_multi_group_layer0_enabled_5periods_v1 \
      runs_graphmask/dbpedia_grouped_20y_v1 \
      runs/dbpedia_grouped_20y_v1 artifacts/dbpedia_grouped_20y_v1 \
      "$CONFIG" config/dbpedia_grouped_rgcn_graphmask_20y_v1.json \
      config/dbpedia_tie_taxonomy_acquired_subgroups_v1.json \
      config/dbpedia_tie_taxonomy_v1.json config/dbpedia_life_periods_20y_v1.json \
      DBpedia/run_layer0_enabled_5periods.sh DBpedia/LAYER0_ENABLED_SUPPLEMENT.md \
      DBpedia/compare_layer0_checkpoint_selection.py DBpedia/grouped_pipeline.py \
      training/graphmask_train.py training/graphmask_report.py
    echo "Created: $REPO/dbpedia_layer0_enabled_5periods_visualization.tar.gz"
    ;;
  *) echo "Usage: $0 [plan|run|summarize|package]" >&2; exit 2 ;;
esac
