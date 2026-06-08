#!/usr/bin/env bash
# Daily pre-lockout refresh.
#
# Runs the full pipeline against the live FIFA Fantasy API and emits a
# recommendation that diffs against yesterday's, including the alternatives
# / sensitivity section.
#
# Usage:
#   ./scripts/daily-snapshot.sh                   # default: GROUP_MD1, no premium boost
#   PREMIUM_BOOST=0.4 ./scripts/daily-snapshot.sh # tilt toward £9M+ players
#   STAGE=GROUP_MD2 ./scripts/daily-snapshot.sh   # other stage (won't compare-to)
#
# Cron-friendly: run from the repo root with the venv activated, or:
#   cd /opt/fifa_wc_fantasy && .venv/bin/python -m … (this script does that)

set -euo pipefail

cd "$(dirname "$0")/.."

VENV_PY=".venv/bin/python"
# Match Python's socket.gethostname() sanitization: strip newline first, then
# replace anything outside the safe alphabet.
HOST="$(hostname | tr -d '\n' | tr -c 'A-Za-z0-9_.-' '_')"
STAGE="${STAGE:-GROUP_MD1}"
PREMIUM_BOOST="${PREMIUM_BOOST:-0.0}"
RESULTS_DIR="results"

YESTERDAY="$(date -u -d 'yesterday' +%Y-%m-%d 2>/dev/null || date -u -v-1d +%Y-%m-%d)"
# Glob covers both legacy date-only filenames and new
# YYYY-MM-DDThh-mm-ssZ timestamped filenames. Pick the lexicographically
# largest (most recent) match if any.
PREVIOUS="$(ls -1 "${RESULTS_DIR}/${HOST}_recommendation_${STAGE}_${YESTERDAY}"*.json 2>/dev/null | sort -r | head -1)"

echo "==> daily snapshot: host=${HOST} stage=${STAGE} premium_boost=${PREMIUM_BOOST}"

"$VENV_PY" -m fifa_fantasy.collector
"$VENV_PY" -m fifa_fantasy.features
"$VENV_PY" -m fifa_fantasy.model --premium-boost "$PREMIUM_BOOST"

OPT_ARGS=(--stage "$STAGE" --report-alternatives)
if [[ -n "$PREVIOUS" && -f "$PREVIOUS" ]]; then
    echo "==> diffing against $PREVIOUS"
    OPT_ARGS+=(--compare-to "$PREVIOUS")
else
    echo "==> no previous snapshot for ${YESTERDAY}; skipping diff"
fi

"$VENV_PY" -m fifa_fantasy.optimizer "${OPT_ARGS[@]}"

echo "==> done"
