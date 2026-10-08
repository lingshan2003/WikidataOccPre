#!/usr/bin/env bash
# Example: bash DBpedia/run_sliding_multi_gpu.sh run all --gpus 4,5
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PYTHON_BIN="${DBPEDIA_PYTHON_BIN:-python}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || [[ -x "$PYTHON_BIN" ]] || {
  echo "Python executable unavailable: $PYTHON_BIN" >&2
  exit 1
}
exec "$PYTHON_BIN" DBpedia/sliding_multi_gpu.py "$@"
