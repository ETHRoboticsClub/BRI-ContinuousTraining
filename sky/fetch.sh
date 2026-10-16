#!/usr/bin/env bash
# Bring the results of a `sky exec` run back: dvc.lock + metrics from the machine, models from S3.
# Usage: sky/fetch.sh [cluster]    (default: hri-gpu)
set -euo pipefail
cluster="${1:-hri-gpu}"
cd "$(dirname "$0")/.."

rsync -av "$cluster:sky_workdir/dvc.lock" .
mkdir -p runs/default
rsync -av "$cluster:sky_workdir/runs/default/metrics.json" "$cluster:sky_workdir/runs/default/history.csv" runs/default/
dvc pull

dvc metrics show
echo
echo "Commit the run with: git add dvc.lock && git commit -m '...' && git push"
