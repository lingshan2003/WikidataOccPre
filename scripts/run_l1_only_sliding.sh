#!/usr/bin/env bash
# Explicit GPU pools only; no experiment is launched without 'run'.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATASET="${1:?Usage: run_l1_only_sliding.sh wikidata|freebase|dbpedia [plan|run] [stage] [options]}"
shift
case "$DATASET" in
  wikidata) PYTHON_BIN="${L1_ONLY_PYTHON_BIN:-${WIKIDATA_PYTHON_BIN:-python}}" ;;
  freebase) PYTHON_BIN="${L1_ONLY_PYTHON_BIN:-${FREEBASE_PYTHON_BIN:-python}}" ;;
  dbpedia) PYTHON_BIN="${L1_ONLY_PYTHON_BIN:-${DBPEDIA_PYTHON_BIN:-python}}" ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac
exec "$PYTHON_BIN" -u scripts/l1_only_sliding.py "$DATASET" "$@"
