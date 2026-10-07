#!/usr/bin/env bash
# Freebase: source adapter -> period graphs -> grouped RGCN -> GraphMask -> summaries.
# Usage: bash Freebase/run_grouped_rgcn_graphmask_20y.sh plan|run [all|prepare|collapse|train|graphmask|summarize] [options]
# Activate wywikidata, then export CUDA_VISIBLE_DEVICES=4 and FREEBASE_GROUP_DEVICE=cuda:0.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
command -v python >/dev/null 2>&1 || {
  echo "python unavailable; activate the wywikidata conda environment first" >&2
  exit 1
}
exec python -u Freebase/grouped_pipeline.py "$@"
