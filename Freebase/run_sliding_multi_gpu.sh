#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v python >/dev/null || { echo "Activate the wywikidata conda environment first." >&2; exit 1; }
exec python -u Freebase/sliding_multi_gpu.py "$@"
