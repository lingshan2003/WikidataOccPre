#!/usr/bin/env bash
# Four windows only. Audit graph construction before allowing pilot training.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PYTHON_BIN="${DBPEDIA_PYTHON_BIN:-python}"
CONFIG="config/dbpedia_grouped_sliding_pilot_4years_v1.json"
PERIODS="center_1900,center_1901,center_1950,center_2000"
MODE="${1:-plan}"
if [[ $# -gt 0 ]]; then shift; fi
# A pilot must not be widened by an earlier shell selection or a CLI override.
for OPTION in "$@"; do
  case "$OPTION" in
    --config|--config=*|--periods|--periods=*|--representations|--representations=*|--include-full)
      echo "Pilot scope is fixed to four years and multi_group; do not pass $OPTION" >&2
      exit 2 ;;
  esac
done
export DBPEDIA_GROUP_INCLUDE_FULL=0
# Retain GPU/path settings; only pin context and representation scope.
case "$MODE" in
  plan)
    exec bash DBpedia/run_sliding_multi_gpu.sh plan all "$@" \
      --config "$CONFIG" --periods "$PERIODS" --representations multi_group
    ;;
  prepare|run)
    bash DBpedia/run_sliding_multi_gpu.sh run prepare "$@" \
      --config "$CONFIG" --periods "$PERIODS" --representations multi_group
    bash DBpedia/run_sliding_multi_gpu.sh run collapse "$@" \
      --config "$CONFIG" --periods "$PERIODS" --representations multi_group
    "$PYTHON_BIN" DBpedia/audit_sliding_window_artifacts.py --config "$CONFIG" --periods "$PERIODS"
    if [[ "$MODE" == "run" ]]; then
      exec bash DBpedia/run_sliding_multi_gpu.sh run all "$@" \
        --config "$CONFIG" --periods "$PERIODS" --representations multi_group
    fi
    ;;
  audit)
    [[ $# -eq 0 ]] || { echo "audit takes no GPU options" >&2; exit 2; }
    exec "$PYTHON_BIN" DBpedia/audit_sliding_window_artifacts.py --config "$CONFIG" --periods "$PERIODS"
    ;;
  *) echo "Usage: $0 plan|prepare|run [--gpus 4,5 | --num-gpus N] ; $0 audit" >&2; exit 2 ;;
esac
