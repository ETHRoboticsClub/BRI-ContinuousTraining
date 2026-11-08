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
src/
  params.yaml          preprocessing and training parameters
  preprocess.py        Preprocess: causal notch + bandpass filter
  dataset.py           EMGFK: continuous recordings sliced into training examples (torch Dataset)
  model/
    tangent_space.py   TangentSpace: sub-window covariances in the Riemannian tangent space
    kernelized.py      KernelizedLinear: linear state-space layer (kernel / stateful modes)
    ssm.py             SSMEncoder: projection + two Kernelized SSM layers, tanh or relu
    model.py           Model: TangentSpace -> SSMEncoder -> dense head, no biases
sky/                   train on a GPU machine with SkyPilot
```

Layer sizes are defined in the model classes. Preprocessing and training parameters live in `src/params.yaml`.

## Model and data

The encoder is two SSM layers trained in kernel (convolutional) mode, the tangent-space covariances being the
input sequence. The model has no biases. `activation` selects `tanh` or `relu`.

The dataset stores each subject's whole recording (`subject_XX.npz`: `emg` [8, T], `angles` [15, T], 200 Hz).
A training example is `raw_length(kernel_length)` raw samples of EMG ending at sample e, with the angles at e as
target. Train examples are `example_stride` raw samples apart; validation and test examples are always 40 apart.

| `train` key | Meaning |
|---|---|
| `k` | Kernel-training example = `k` stateful inputs (240 samples each), capped by the shortest recording |
| `example_stride` | Raw samples between consecutive training examples |
| `activation` | `tanh` or `relu` |
| `seeds` | One model is trained per seed; metrics are reported as mean and sample std over seeds |

## Run

```bash
dvc repro                  # build the dataset (downloads ~2 GB on first run) and train
dvc metrics show           # val and test MSE / R²
dvc plots show             # training curves
dvc push                   # share the built dataset and output/
```

or without DVC: `python scripts/build_dataset.py` then `python scripts/train.py`.

## Data versioning

Raw recordings are downloaded from Zenodo and not versioned. DVC versions the built dataset
(`data/processed/emg_fk`) and the training output (`output/`) through `dvc.lock`, so every commit
points to the exact windows and model it trained.

`output/` holds `seed_<n>/best_model.pt` per seed, `history.csv` (all seeds, with a `seed` column) and
`metrics.json` (`kernel_length`, `example_length`, `val`/`test` mean and std of mse, r2, nmse, mae in degrees, and
per-seed values).

Splits are by subject (`train.val_subjects`, `train.test_subjects`), never random windows, because windows overlap.
