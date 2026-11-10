# HRI Continuous Training — Intra-Subject GRU Decoder

Decode 15 finger joint angles from 8-channel 200 Hz forearm EMG with an intra-subject
Temporal Riemannian Regressor (Colot et al., 2026) combining 3-band Riemannian tangent-space
covariances and phasic/tonic envelope features with a 2-layer GRU decoder.

The repo holds **one model and one approach per commit**. To try something else, create a branch.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Structure

```
dvc.yaml               pipeline: build_dataset -> train
scripts/
  build_dataset.py     extract 3-band Riemannian tangent + envelope sequences to data/processed/emg_fk
  train.py             train intra-subject GRU decoder -> output/best_model.pt
src/
  params.yaml          preprocessing and training parameters
  preprocess.py        Preprocess: causal notch + 3-band Butterworth filter bank
  dataset.py           EMGFK: torch Dataset wrapper
  trainer.py           train_epoch, evaluate, fit, save_history
  model/
    tangent_space.py   TangentSpace, ledoit_wolf_cov, riemannian_whitening
    gru.py             GRUEncoder: dense projection + stacked GRUs
    model.py           Model: TangentSpace -> GRUEncoder -> dense head
sky/                   train on a GPU machine with SkyPilot
```

## Run

```bash
python scripts/build_dataset.py
python scripts/train.py --data data/recordings/myo_intrasubject.npz --out output/best_model.pt
```
