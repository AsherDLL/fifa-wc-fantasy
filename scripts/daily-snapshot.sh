#!/usr/bin/env bash
# Daily pipeline refresh.
#
# Chains collector -> features -> model -> optimizer with sensible
# defaults. Writes per-run results to ./results/ with hostname prefix and
# UTC timestamp suffix so reruns coexist.
#
# Usage:
#   ./scripts/daily-snapshot.sh                   # default: GROUP_MD1, no premium boost
#   PREMIUM_BOOST=0.4 ./scripts/daily-snapshot.sh # tilt toward $9M+ players
#   STAGE=GROUP_MD2 ./scripts/daily-snapshot.sh   # other stage

set -euo pipefail

cd "$(dirname "$0")/.."

VENV_PY=".venv/bin/python"
STAGE="${STAGE:-GROUP_MD1}"
PREMIUM_BOOST="${PREMIUM_BOOST:-0.0}"

"$VENV_PY" -m fifa_fantasy.collector
"$VENV_PY" -m fifa_fantasy.features
"$VENV_PY" -m fifa_fantasy.model --premium-boost "$PREMIUM_BOOST"
"$VENV_PY" -m fifa_fantasy.optimizer --stage "$STAGE"
