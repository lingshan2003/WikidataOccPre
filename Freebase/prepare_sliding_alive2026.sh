#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v python >/dev/null || { echo "Activate the wywikidata conda environment first." >&2; exit 1; }
LOG_ROOT="artifacts/freebase_alive2026_construction_logs"
mkdir -p "$LOG_ROOT"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
export FREEBASE_CONSTRUCTION_LOG="$ROOT/$LOG_ROOT/$RUN_ID.log"
set +e
python -u Freebase/prepare_sliding_alive2026.py "$@" 2>&1 | tee "$FREEBASE_CONSTRUCTION_LOG"
EXIT_CODE=${PIPESTATUS[0]}
set -e
echo "$EXIT_CODE" > "$LOG_ROOT/$RUN_ID.exit_code"
exit "$EXIT_CODE"
