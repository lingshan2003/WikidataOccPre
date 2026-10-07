#!/usr/bin/env bash
# Usage: CBDB_PYTHON_BIN=python bash CBDB/run_pipeline.sh plan|run [stage] [options]
# Select a free physical GPU first: export CUDA_VISIBLE_DEVICES=2
# Then --device cuda:0 refers to that selected GPU; the environment is inherited.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PYTHON_BIN="${CBDB_PYTHON_BIN:-python}"
exec "$PYTHON_BIN" CBDB/pipeline.py "$@"
