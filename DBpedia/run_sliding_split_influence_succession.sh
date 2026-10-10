#!/usr/bin/env bash
# Eight-group rerun with the explicit born-after-1920 / alive-through-2026 rule.
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
CONFIG="config/dbpedia_grouped_sliding_1900_2000_pm20_step1_split_influence_succession_alive2026_v3.json"
NAME="dbpedia_grouped_sliding_1900_2000_pm20_step1_split_influence_succession_alive2026_v3"
# Pin the experiment identity even when the shell still has v1 path overrides.
export DBPEDIA_GROUP_SOURCE_DATA="artifacts/dbpedia_2022_priority_v1/graph_data.pt"
export DBPEDIA_GROUP_PERIOD_CONFIG="config/dbpedia_life_windows_1900_2000_pm20_step1_alive2026_v2.json"
export DBPEDIA_GROUP_PERIOD_ROOT="artifacts/dbpedia_life_windows_1900_2000_pm20_step1_alive2026_v2"
export DBPEDIA_GROUP_BINARY_TAXONOMY="config/dbpedia_tie_taxonomy_v1.json"
export DBPEDIA_GROUP_MULTI_GROUP_TAXONOMY="config/dbpedia_tie_taxonomy_acquired_subgroups_v2.json"
export DBPEDIA_GROUP_RELATION_ROOT="artifacts/$NAME"
export DBPEDIA_GROUP_MODEL_ROOT="runs/$NAME"
export DBPEDIA_GROUP_GRAPHMASK_ROOT="runs_graphmask/$NAME"
export DBPEDIA_GROUP_INCLUDE_FULL=0
export DBPEDIA_GROUP_SEED=42
for OPTION in "$@"; do
  case "$OPTION" in
    --config|--config=*|--periods|--periods=*|--representations|--representations=*|--include-full)
      echo "This wrapper pins the 101-window eight-group experiment; use run_sliding_multi_gpu.sh directly for custom scope: $OPTION" >&2
      exit 2 ;;
  esac
done
exec bash DBpedia/run_sliding_multi_gpu.sh "$@" \
  --config "$CONFIG" --periods all --representations multi_group
