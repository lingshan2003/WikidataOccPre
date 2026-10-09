#!/usr/bin/env bash
# BHHT-reviewed v2 labels; activate wywikidata and explicitly export the device.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FREEBASE_GROUP_CONFIG="config/freebase_grouped_rgcn_graphmask_20y_v2.json"
exec bash "$REPO/Freebase/run_grouped_rgcn_graphmask_20y.sh" "$@"
