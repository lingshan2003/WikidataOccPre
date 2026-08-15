#!/usr/bin/env bash
# Level-2-in / Level-2-out life-period inherited/acquired tie audit.
#
# This wrapper deliberately uses separate artifacts and reports from the
# Level-1 audit.  Its source artifact must predict occupation_level2, and the
# model is given only Level-2 occupation features (held-out nodes remain
# masked under the shared training protocol).

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export RGCN_PERIOD_SOURCE_DATA="artifacts/level2_hierarchy/graph_data.pt"
export RGCN_PERIOD_ARTIFACT_ROOT="artifacts/level2_life_period_induced_v2"
export RGCN_PERIOD_AUDIT_OUTPUT_ROOT="runs_report/level2/life_period_induced_tie_audit_v2"
export RGCN_PERIOD_OCCUPATION_FEATURE_LEVELS="2"
export RGCN_PERIOD_EXPECT_TARGET_COLUMN="occupation_level2"

exec bash "$PROJECT_DIR/scripts/run_period_induced_tie_audit_experiments.sh" "$@"
