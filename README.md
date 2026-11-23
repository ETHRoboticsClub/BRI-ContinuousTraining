# BRI-TrainingPipeline — WaveFormer 6-Class sEMG Classifier

Training and offline evaluation pipeline for the **WaveFormer** 6-class sEMG gesture recognition model (*Chen, Orlandi, Rapa, Benatti, Benini, Li — arXiv:2506.11168v3*, `distinct-classifier-WaveFormer` branch).

## Architecture Highlights
- **Input**: Signed bipolar 8-channel 200 Hz Myo sEMG windows `(B, 1, 8, 200)` (1.0 s), DC-centered per window and scaled by `/ 128.0`.
- **PatchEmbed**: 2D patch projection `(1, 20)` -> `embed_dim=256` with LayerNorm + GELU (`8 x 10 = 80` spatio-temporal patch tokens + `[CLS]`).
- **WaveletConv (`WTConv2d`)**: 3-level learnable 2D `db1` Haar wavelet decomposition/reconstruction with depthwise `3x3` subband convolutions, high-frequency subband dropout (`0.1`), and pointwise residual channel mixing.
- **RoPE Transformer Encoder**: 6-layer encoder with 8-head Rotary Positional Embeddings (`RoPEAttention`), 2D channel/temporal embeddings, DropPath (`0.1`), and `cls_token` classification head (`~3.28M` params).

## 6-Class Taxonomy

| Class ID | Label | Role |
| :---: | :--- | :--- |
| `0` | `Relax` | Neutral / rest |
| `1` | `Open` | Hand extension |
| `2` | `Close` | Power grasp |
| `3` | `Wave-In` | Wrist flexion |
| `4` | `Wave-Out` | Wrist extension |
| `5` | `Pinch` | Precision thumb–index pinch |

## Training & Evaluation

Run inside `C:\Users\pette\Documents\ETHRC_HRI\BRI-TrainingPipeline`:

```powershell
# Train the 6-class WaveFormer model on GPU
.venv\Scripts\python.exe scripts/train.py --config src/params.yaml --device cuda

# Evaluate validation & causal 10 Hz streaming trial accuracy
.venv\Scripts\python.exe scripts/evaluate.py --config src/params.yaml
```

Outputs are written to `output/best_model.pt`, `output/history.csv`, and `output/metrics.json`.

## Testing

```powershell
.venv\Scripts\python.exe -m pytest -v
```
