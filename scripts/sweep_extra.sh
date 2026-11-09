#!/usr/bin/env bash
# Extra experiments (augmentation, normalization, SSM regularization), one change at a time, as DVC experiments
# named x-<activation>-k<K>-s<STRIDE>-<label>, each compared with a baseline rerun of the same base configuration.
# Usage: scripts/sweep_extra.sh tanh|relu K STRIDE    -> extra_<activation>_k<K>_s<STRIDE>.csv
#                                                        (summarize with scripts/summarize_sweep.py)
# BASE_SETS="key=value ..." adds base overrides of src/params.yaml train keys to every run (e.g. Section 4
# winners); a key that one of the changes, or the arguments, also set is refused.
# DRY_RUN=1 prints the dvc exp run commands and runs nothing.
# Set PUSH_EXPERIMENTS=1 to also run `dvc exp push` to the git remote (default: off).
# An existing experiment name makes dvc exp run fail: remove old runs with `dvc exp remove` first.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ "$#" -ne 3 ] || { [ "$1" != "tanh" ] && [ "$1" != "relu" ]; }; then
  echo "usage: $0 tanh|relu K STRIDE" >&2
  exit 1
fi
activation="$1"
k="$2"
stride="$3"

# "label:change"
changes=(
  "baseline:"
  "aug_gain:aug_gain=0.2"
  "aug_rotate:aug_rotate=1"
  "aug_shift:aug_shift=0.5"
  "norm_set:normalization=set"
  "downsample_2:downsample=2"
  "downsample_5:downsample=5"
  "speed_1:ssm_speed=1"
  "speed_0p25:ssm_speed=0.25"
  "ssm_lr_0p1:ssm_lr_scale=0.1"
  "no_decay_a:ssm_decay_A=false"
)
reserved="activation k example_stride aug_gain aug_rotate aug_shift normalization downsample ssm_speed ssm_lr_scale ssm_decay_A"

base=(
  -S "src/params.yaml:train.activation=$activation"
  -S "src/params.yaml:train.k=$k"
  -S "src/params.yaml:train.example_stride=$stride"
)
read -ra base_sets <<< "${BASE_SETS:-}"
for set in ${base_sets[@]+"${base_sets[@]}"}; do
  key="${set%%=*}"
  for r in $reserved; do
    if [ "$key" = "$r" ]; then
      echo "BASE_SETS may not set $key: the arguments or one of the changes set it" >&2
      exit 1
    fi
  done
  base+=(-S "src/params.yaml:train.$set")
done

dry_run="${DRY_RUN:-0}"
if [ "$dry_run" != "1" ]; then
  dvc pull --allow-missing
  dvc repro build_dataset
  dvc push
fi

for entry in "${changes[@]}"; do
  label="${entry%%:*}"
  change="${entry#*:}"
  cmd=(dvc exp run -n "x-$activation-k$k-s$stride-$label" "${base[@]}")
  if [ -n "$change" ]; then
    cmd+=(-S "src/params.yaml:train.$change")
  fi
  if [ "$dry_run" = "1" ]; then
    echo "${cmd[*]}"
  else
    "${cmd[@]}"
  fi
done
if [ "$dry_run" = "1" ]; then
  exit 0
fi

if [ "${PUSH_EXPERIMENTS:-0}" = "1" ] && [ -n "$(git remote)" ]; then
  dvc exp push "$(git remote | head -n 1)" || echo "dvc exp push failed, continuing" >&2
fi
dvc exp show --csv > "extra_${activation}_k${k}_s${stride}.csv"
