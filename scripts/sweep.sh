#!/usr/bin/env bash
# k x example_stride sweep as DVC experiments, for one activation.
# Usage: scripts/sweep.sh tanh|relu    -> sweep_<activation>.csv (summarize with scripts/summarize_sweep.py)
# Set PUSH_EXPERIMENTS=1 to also run `dvc exp push` to the git remote (default: off).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -ne 1 ] || { [ "$1" != "tanh" ] && [ "$1" != "relu" ]; }; then
  echo "usage: $0 tanh|relu" >&2
  exit 1
fi
activation="$1"

dvc pull --allow-missing
dvc repro build_dataset
dvc push

# "k stride" pairs
configs=(
  "1 40" "1 120" "1 240"
  "5 40" "5 240" "5 520" "5 1040"
)
for config in "${configs[@]}"; do
  read -r k stride <<< "$config"
  dvc exp run \
    -S "src/params.yaml:train.activation=$activation" \
    -S "src/params.yaml:train.k=$k" \
    -S "src/params.yaml:train.example_stride=$stride"
done

if [ "${PUSH_EXPERIMENTS:-0}" = "1" ] && [ -n "$(git remote)" ]; then
  dvc exp push "$(git remote | head -n 1)" || echo "dvc exp push failed, continuing" >&2
fi
dvc exp show --csv > "sweep_${activation}.csv"
