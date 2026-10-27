#!/usr/bin/env bash
# Bring the results of a `sky exec` run back: dvc.lock from the machine, outputs from S3.
# Usage: sky/fetch.sh [cluster]    (default: hri-gpu)
set -euo pipefail
cluster="${1:-hri-gpu}"
cd "$(dirname "$0")/.."

rsync -av "$cluster:sky_workdir/dvc.lock" .
dvc pull train

dvc metrics show
echo
echo "Commit the run with: git add dvc.lock && git commit -m '...' && git push"
