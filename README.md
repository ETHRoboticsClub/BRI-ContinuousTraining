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
  build_dataset.py     download EMG-FK, preprocess, save windows to data/processed/emg_fk
  train.py             load data, load model, train -> output/
src/
  params.yaml          preprocessing and training parameters
  preprocess.py        Preprocess: causal notch + bandpass filter
  dataset.py           EMGFK: download + windowing (build) and torch Dataset
  model/
    tangent_space.py   TangentSpace: sub-window covariances in the Riemannian tangent space
    gru.py             GRUEncoder: dense projection + stacked GRUs
    model.py           Model: TangentSpace -> GRUEncoder -> dense head
sky/                   train on a GPU machine with SkyPilot
```

Dataset and model settings (window, stride, subjects, layer sizes) are defined in their classes.
Only preprocessing and training parameters live in `src/params.yaml`.

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

Splits are by subject (`train.val_subjects`, `train.test_subjects`), never random windows, because windows overlap.
