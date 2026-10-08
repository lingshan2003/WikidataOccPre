#!/usr/bin/env bash
# Annual centered windows, independent RGCN + GraphMask per selected window.
# Same options/stages as run_grouped_rgcn_graphmask_20y.sh.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
export DBPEDIA_GROUP_CONFIG="${DBPEDIA_SLIDING_CONFIG:-config/dbpedia_grouped_sliding_1900_2000_pm20_step1_v1.json}"
exec bash DBpedia/run_grouped_rgcn_graphmask_20y.sh "$@"
