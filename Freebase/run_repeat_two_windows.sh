#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "${FREEBASE_PYTHON_BIN:-python}" -u Freebase/repeat_two_windows.py "$@"
