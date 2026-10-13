#!/usr/bin/env bash
set -Eeuo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${WIKIDATA_PYTHON_BIN:-python}"
if [[ "${1:-plan}" != "run" ]]; then
  exec "$PYTHON_BIN" Wikidata/sliding_multi_gpu.py "$@"
fi
LOG_DIR="${WIKIDATA_RUN_LOG_DIR:-artifacts/wikidata_sliding_run_logs}"
[[ "$LOG_DIR" = /* ]] || LOG_DIR="$PROJECT_DIR/$LOG_DIR"
mkdir -p "$LOG_DIR"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
export WIKIDATA_RUN_LOG="$LOG_DIR/$RUN_ID.log"
printf 'Log: %s\n' "$WIKIDATA_RUN_LOG"
set +e
"$PYTHON_BIN" Wikidata/sliding_multi_gpu.py "$@" 2>&1 | tee "$WIKIDATA_RUN_LOG"
RUN_STATUS=${PIPESTATUS[0]}
printf '%s\n' "$RUN_STATUS" > "$LOG_DIR/$RUN_ID.exit_code"
exit "$RUN_STATUS"
