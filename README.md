# Training Pipeline (Embedding, Discrete Decoder, and Continuous Decoder)

Train models that map forearm EMG to robotic arm actuator activations.

The repo holds **one model at a time**. To try something else, create a branch, change
`src/model.py` (and `src/preprocessing.py` if needed), and keep it up to date with
`git rebase origin/main`. Everything else (dataset, trainer, metrics, probes) is shared,
so branches stay easy to compare and to merge.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
dvc pull
```

## Structure

```
params.yaml            all parameters
dvc.yaml               pipeline: build_dataset -> train
scripts/
  build_dataset.py     raw recordings -> windows
  train.py             your training script
src/
  preprocessing.py     preprocess(emg, fs, ...): change it if your method needs it
  model.py             Model: your model
  dataset.py           recordings, windowing, train/val/test splits, loaders
  trainer.py           train / evaluate (shared, do not fork)
  metrics.py           regression metrics (shared)
  probes.py            @probe decorators to monitor layers (shared)
tests/                 end-to-end test on fake data
```

## Run

```bash
dvc repro                         # build the dataset and train, outputs in runs/default/
dvc metrics show                  # val and test metrics
dvc plots show                    # training curves
dvc exp run -S train.lr=3e-4      # try another parameter value, compare with: dvc exp show
```

or without DVC: `python scripts/build_dataset.py` then `python scripts/train.py`.

## Data

Each recording is one `.npz` file in `data/raw/<name>/`:

| key             | shape    | content                                          |
|-----------------|----------|--------------------------------------------------|
| `emg`           | [C, T]   | raw EMG                                          |
| `fs`            | scalar   | sampling rate (Hz)                               |
| `channel_names` | [C]      | electrode names                                  |
| `targets`       | [D, T]   | actuator activations in [0, 1], aligned with emg |
| `target_names`  | [D]      | actuator names                                   |
| `meta_<key>`    | scalar   | int, for every key in `dataset.meta_keys`        |

Add new data with `dvc add data/raw/<name> && git add data/raw/<name>.dvc && dvc push`.

### NinaPro DB2

The current dataset is [NinaPro DB2](https://ninapro.hevs.ch/instructions/DB2.html), subjects 1-10, converted by
`scripts/fetch_ninapro.py` (`dvc pull` is enough to get it):

- `emg`: 12 Delsys electrodes at 2 kHz.
- `targets`: the 22 CyberGlove sensors, scaled per subject to [0, 1] between their 1st and 99th percentiles.
- One file per movement repetition (`data/raw/ninapro_db2/s<subject>/e<exercise>_m<movement>_r<repetition>.npz`),
  exercises 1 and 2 only (exercise 3 has no glove).
- Test set: repetitions 2 and 5 (the standard NinaPro split), validation: repetition 4.

More subjects: `python scripts/fetch_ninapro.py --subjects 11-40`, then `dvc add data/raw/ninapro_db2` and `dvc push`.
Splits use the metadata (`dataset.test`, `train.val`), never random windows, because windows overlap.

## Changing the model

1. Edit `Model` in `src/model.py`. It receives `(in_channels, out_dim, **params["model"])`
   and maps `[batch, channels, window]` to `[batch, n_targets]`. Use only past samples.
2. Put its hyperparameters in the `model:` section of `params.yaml`.
3. Adapt `scripts/train.py` if needed (loss, optimizer...), always calling `src.trainer.train`.
4. Run `pytest`.

## Probes

Decorate the methods of your model whose outputs you want to monitor. The trainer computes
the metrics on a validation subset every `every` epochs and on the best model:

```python
from functools import partial
from src.probes import probe, saturation


@probe("embedding", "linear_probe", "effective_rank", "dead_units")
@probe("embedding", "tsne", every=5)
def embed(self, x): ...


@probe("hidden", partial(saturation, low=-1, high=1))  # metric with options
def recur(self, x): ...
```

| layer            | metrics                                                                |
|------------------|------------------------------------------------------------------------|
| any              | `activation_stats`, `dead_units`, `effective_rank`, `linear_probe`, `target_cka`, `pca`, `tsne` |
| convolution      | `channel_activity`, `partial(dead_units, unit_axis=1)`, `partial(temporal_dynamics, time_axis=-1)` |
| recurrent        | `temporal_dynamics`, `partial(saturation, low=-1, high=1)`             |
| sigmoid / output | `saturation`                                                           |
| attention        | `attention_entropy`                                                    |

A custom metric is any function `(values, targets) -> dict` (or a matplotlib `Figure`).
Results go to `runs/<name>/probes/`: `scalars.csv`, one plot per probe, and the figures.

## Metrics

`r2`, `mae`, `mse`, `pearson` and `jitter_ratio` (how much the prediction moves compared to
the target: above 1, the arm shakes). Compare models on the `test` metrics of `metrics.json`.
