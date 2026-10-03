#!/usr/bin/env bash
# DBpedia: 8 periods x (binary, multi_group), independent RGCN + GraphMask.
# Usage: bash DBpedia/run_grouped_rgcn_graphmask_20y.sh plan|run [all|prepare|collapse|train|graphmask|summarize] [options]
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PYTHON_BIN="${DBPEDIA_PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] || {
  echo "Python executable unavailable: $PYTHON_BIN" >&2
  exit 1
}
exec "$PYTHON_BIN" DBpedia/grouped_pipeline.py "$@"
