# BRI-TrainingPipeline — ReactEMG 6-Class Masked Multimodal Classifier

Training and offline evaluation pipeline for the **ReactEMG** (`Any2Any_Model`) 6-class sEMG intent classifier (*Wang, Zhu et al., arXiv:2506.19815*).

## 6-Class Taxonomy

| Class ID | Label | Mask / Role |
| :---: | :--- | :--- |
| `0` | `Relax` | Neutral / rest |
| `1` | `Open` | Hand extension |
| `2` | `Close` | Power grasp |
| `3` | `Wave-In` | Wrist flexion |
| `4` | `Wave-Out` | Wrist extension |
| `5` | `Pinch` | Precision thumb–index pinch |
| `6` | `<MASK>` | Multimodal query token |

## Setup & Training

Run inside `C:\Users\pette\Documents\ETHRC_HRI\BRI-TrainingPipeline`:

```powershell
# 1. Download & unpack the official preprocessed ReactEMG archive (data.zip)
.venv\Scripts\python.exe scripts/build_dataset.py

# 2. Train the 6-class Any2Any_Model from scratch on CUDA
.venv\Scripts\python.exe scripts/train.py

# 3. Evaluate zero-lookahead (l=0) and smoothed look-ahead (l=50, s=20)
.venv\Scripts\python.exe scripts/evaluate.py
```

Outputs are written to `output/best_model.pt`, `output/history.csv`, and `output/metrics.json`.

## Testing

```powershell
.venv\Scripts\python.exe -m pytest -v
```
