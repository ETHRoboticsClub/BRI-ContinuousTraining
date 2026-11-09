# HRI Continuous Training

Decode 15 finger joint angles from 8-channel forearm EMG with a Temporal Riemannian Regressor
(Colot et al., 2026), trained cross-subject on the EMG-FK dataset.

The repo holds **one model and one approach per commit**. To try something else, create a branch.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
dvc pull            # built dataset + trained model of this commit
```

S3 access for DVC is described in [docs/dvc-access.md](docs/dvc-access.md).

## Structure

```
dvc.yaml               pipeline: build_dataset -> train
scripts/
  build_dataset.py     download EMG-FK, preprocess, save whole recordings to data/processed/emg_fk
  train.py             load data, load model, train -> output/
  sweep.sh             k x example_stride sweep as DVC experiments, one activation
  sweep_extra.sh       augmentation / normalization / SSM regularization changes, one at a time, on one base
  sweep_extra_local.py the same without DVC, resumable after the machine turns off
  summarize_sweep.py   pick the configuration with the lowest validation MAE, compare test MAE with the paper
src/
  params.yaml          preprocessing and training parameters
  preprocess.py        Preprocess: causal notch + bandpass filter
  augment.py           Augment: random per-example channel gain, electrode-ring rotation and shift (training only)
  dataset.py           EMGFK: continuous recordings sliced into training examples (torch Dataset)
  model/
    tangent_space.py   TangentSpace: sub-window covariances in the Riemannian tangent space
    kernelized.py      zoh_discretize, Kernelized (diagonal complex SSM, FFT conv), KernelizedLinear
    ssm.py             SSMEncoder: projection + two Kernelized SSM layers, tanh or relu
    model.py           Model: TangentSpace -> SSMEncoder -> dense head, no biases
sky/                   train on a GPU machine with SkyPilot
```

Layer sizes are defined in the model classes. Preprocessing and training parameters live in `src/params.yaml`.

## Model and data

The encoder is two SSM layers trained in kernel (convolutional) mode, the tangent-space covariances being the
input sequence. The model has no biases. `activation` selects `tanh` or `relu`.

The dataset stores each subject's whole recording (`subject_XX.npz`: `emg` [8, T], `angles` [15, T], 200 Hz). The
EMG is stored filtered but not normalized; `EMGFK` z-scores each channel when it loads the recordings.
A training example is `raw_length(kernel_length)` raw samples of EMG ending at sample e, with the angles at e as
target. Train examples are `example_stride` raw samples apart; validation and test examples are always 40 apart.

| `train` key | Meaning |
|---|---|
| `k` | Kernel-training example = `k` stateful inputs (240 samples each), capped by the shortest recording (over all subjects) |
| `example_stride` | Raw samples between consecutive training examples |
| `activation` | `tanh` or `relu` |
| `normalization` | EMG z-score per channel: `subject` (each recording on its own statistics), `set` (statistics pooled over the recordings of each split, so only training changes: validation and test are one subject each) or `none` |
| `aug_gain` | Training examples only: each EMG channel is scaled by `exp(N(0, aug_gain))`, drawn per example; 0 = off |
| `aug_rotate` | Training examples only: the 8 channels are rotated around the electrode ring by r ~ U{-aug_rotate, ..., aug_rotate} electrodes, drawn per example; 0 = off |
| `aug_shift` | Training examples only: fractional rotation by u ~ U(-aug_shift, aug_shift) electrodes, linearly interpolating between ring neighbours, drawn per example; 0 = off |
| `downsample` | Keep every s-th tangent step after the first SSM layer (MetaTF's `Stride`); the second SSM layer sees L / s steps. s must be a power of 2, divide L and leave at least 2 steps (so 4 needs k even: L = 10k) |
| `ssm_speed` | `speed` of both Kernelized layers (MetaTF default 4): their initial dt is proportional to it; dt stays trainable |
| `ssm_lr_scale` | The SSM dynamics (`A`, `log_dt` of both Kernelized layers) train at `lr` x this (own AdamW parameter group) |
| `ssm_decay_A` | `false`: no weight decay on `A` |
| `seeds` | One model is trained per seed; metrics are reported as mean and sample std over seeds |

## Run

```bash
dvc repro                  # build the dataset (downloads ~2 GB on first run) and train
dvc metrics show           # val and test MAE (degrees), NMSE, MSE, R²: mean/std over seeds
dvc plots show             # training curves
dvc push                   # share the built dataset and output/
```

or without DVC: `python scripts/build_dataset.py` then `python scripts/train.py`.

## Sweep

`scripts/sweep.sh tanh|relu` builds the dataset, then runs one `dvc exp run` per configuration (k=1 with stride
40, 120, 240; k=5 with stride 40, 240, 520, 1040) and writes `sweep_<activation>.csv` (`dvc exp show --csv`).
It does not push experiment refs unless `PUSH_EXPERIMENTS=1` is set.
On the GPU machine:

```bash
sky exec hri-gpu sky/train.yaml --env SWEEP=tanh   # then SWEEP=relu
rsync -av hri-gpu:sky_workdir/sweep_tanh.csv .
python scripts/summarize_sweep.py sweep_tanh.csv
```

The summary lists val and test MAE of every configuration and selects the one with the lowest **validation** MAE
(test is never used for selection). Verdict on its test MAE against the paper's 16.71 degrees: `success` below
16.71, `threshold` up to 17.21, otherwise `miss`.

### Extra experiments

`scripts/sweep_extra.sh tanh|relu K STRIDE` runs 11 experiments on one base configuration: a baseline rerun and
one experiment per change, in this order: SSM regularization (`downsample=2` and `4`, `ssm_speed=1` and `0.25`,
`ssm_lr_scale=0.1`, `ssm_decay_A=false`), normalization (`normalization=set`), then augmentation (`aug_gain=0.2`,
`aug_rotate=1`, `aug_shift=0.5`). Each is a DVC experiment named
`x-<activation>-k<K>-s<STRIDE>-<change>`. `BASE_SETS="key=value ..."` adds base overrides to every run (it refuses
keys that a change sets); `DRY_RUN=1` only prints the commands. Remove old runs with `dvc exp remove` before
rerunning the same base.

```bash
sky exec hri-gpu sky/train.yaml --env EXTRA="tanh 5 40" --env BASE_SETS=""
rsync -av hri-gpu:sky_workdir/extra_tanh_k5_s40.csv .
python scripts/summarize_sweep.py extra_tanh_k5_s40.csv
```

On a laptop without DVC, `python -m scripts.sweep_extra_local relu 10 40 [--start downsample_2]` runs the same 11
experiments (from `--start` on) into `output/extra_<activation>_k<K>_s<STRIDE>/<change>/`. Rerunning the same
command resumes: finished experiments and finished seeds are skipped, so a shutdown costs at most the seed in
progress. While it runs, Windows does not sleep on idle (closing the lid or shutting down still stops it).
Summarize with `python scripts/summarize_sweep.py output/extra_relu_k10_s40 --baseline output/sweep_relu/k10_s40`
(`--baseline` when the baseline was not rerun).

The summary prints each change's val MAE (mean ± seed std) and test MAE with their differences from the baseline,
marks the changes whose val MAE is below the baseline's by more than its seed std, and selects the change with the
lowest **validation** MAE.

## Data versioning

Raw recordings are downloaded from Zenodo and not versioned. DVC versions the built dataset
(`data/processed/emg_fk`) and the training output (`output/`) through `dvc.lock`, so every commit
points to the exact recordings and model it trained.

`output/` holds `seed_<n>/best_model.pt` per seed, `history.csv` (all seeds, with a `seed` column) and
`metrics.json` (`kernel_length`, `example_length`, `val`/`test` mean and std of mse, r2, nmse, mae in degrees, and
per-seed values; std is 0 when only one seed is run).

Splits are by subject (`train.val_subjects`, `train.test_subjects`), never random training examples, because examples overlap.
