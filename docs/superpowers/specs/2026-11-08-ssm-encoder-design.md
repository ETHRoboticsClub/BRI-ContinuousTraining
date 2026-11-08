# SSM encoder for TRR: replace the GRUs with Kernelized SSM layers

Status: draft for review · Branch: `ssm` · Date: 2026-11-08

## 1. Intent

The current model is the Temporal Riemannian Regressor (TRR, Colot et al., 2026): tangent-space covariance
features feed a dense layer and two stacked GRUs, then a dense head. We want the same model with the GRU
encoder and every linear layer replaced by **Kernelized state-space (SSM) layers** as defined in BrainChip's
MetaTF (`akida_models`).

Why: the long-term goal is running this decoder on **Akida hardware**. MetaTF's Kernelized layers train as
convolutions over the whole sequence (fast, parallel) and convert to a stateful recurrent layer for streaming
on Akida. Getting a PyTorch version that trains well, and matches MetaTF exactly, is the first step.

Only the model architecture should change. The dataset changes only where the new architecture forces it
(longer training examples, Section 5).

## 2. Goals, non-goals, anti-goals

**Goals**
- A PyTorch re-implementation of MetaTF's `Kernelized` layer that matches the original numerically (forward
  and gradients), proven by tests against reference outputs from the real MetaTF layer.
- Replace `GRUEncoder` and all `nn.Linear` layers with SSM layers, keeping every layer's input/output width.
- Train in kernel (convolutional) mode; validate and test in kernel mode.
- Sweep the kernel multiplier k and the example stride (Section 7) and compare against the paper.

**Non-goals (this round)**
- Stateful / streaming inference, `convert_to_stateful`, Akida conversion or quantization.
- Changing the front end (`TangentSpace`), preprocessing, split, loss, optimizer or schedule.
- Keeping the GRU available. It is replaced outright.

**Anti-goals (a failure even if it trains)**
- A layer that only resembles MetaTF's `Kernelized`. If it doesn't match MetaTF numerically, a later port to
  Akida is not trustworthy.
- Test-set leakage: choosing the configuration by its test score.

## 3. Terms

All sizes are at 200 Hz. "Window" alone is never used; one of these names is.

| Term | Size | Meaning |
|---|---|---|
| Raw sample | 5 ms | One EMG reading |
| Sub-window | 60 raw samples (300 ms), step 20 raw samples (100 ms) | Becomes one covariance → one tangent vector (36 values for 8 channels) |
| Tangent step | 1 tangent vector | One time step of the sequence the SSM layers see |
| Stateful input size | 10 tangent steps = 240 raw samples (1.2 s) | Today's encoder input; the chunk size for future stateful streaming |
| Kernel length L | `L = k × 10` tangent steps | Kernel training only: the SSM kernel has one tap per tangent step of the input |
| Training example | `60 + (L − 1) × 20 = 200k + 40` raw samples | One slice of raw EMG fed to the model in kernel mode. k=1 → 240, k=5 → 1040 |

**Why a training example is not `k × 240` raw samples.** k multiplies the stateful input size counted in
tangent steps (10), because the SSM layers only ever see tangent steps. Sub-windows overlap (60 long, a new one
every 20), so each extra tangent step adds 20 raw samples, not 24. 50 tangent steps therefore span 1040 raw
samples. Five back-to-back 240-raw-sample chunks (1200 raw samples) would instead hold 58 tangent steps.
| Target | 15 joint angles | Angles at the last raw sample of the training example |
| Example stride | raw samples | Distance between the end samples of consecutive training examples. Today 40 |

**Kernel mode vs stateful mode.** In MetaTF the `Kernelized` layer has no kernel-size setting: on each forward
pass it builds a kernel with exactly as many taps as the input has time steps, and convolves causally. So the
kernel length used in training equals the training-example length in tangent steps. A stateful model streams
chunks of the stateful input size and carries state between them. MetaTF's UORED example does exactly this:
it trains on 8192-sample inputs and streams in 256-sample chunks (k = 32). Stateful mode is out of scope here.

## 4. Architecture

```
EMG training example [B, 8, 60+(L−1)·20]
  → TangentSpace (unchanged; no trainable params)          [B, L, 36]
  → KernelizedLinear 36 → 256 → act                         [B, L, 256]
  → Kernelized 256 → 256 (N = 64) → act → Dropout(0.1)      [B, L, 256]
  → Kernelized 256 → 128 (N = 64) → act                     [B, L, 128]
  → last tangent step → Dropout(0.1)                        [B, 128]
  → KernelizedLinear 128 → 64 → act                         [B, 64]
  → KernelizedLinear 64 → 15                                [B, 15] joint angles
```

This mirrors the current model layer for layer: dense+Tanh → GRU → dropout → GRU → last step → dropout →
dense+Tanh → dense. **No layer has a bias** (decided). `build_model` no longer initializes an output bias to
the mean target.

**Activation after each Kernelized layer (decided).** A GRU is nonlinear inside; a Kernelized layer is linear.
Without an activation between them, the two Kernelized layers would collapse into one linear filter. MetaTF's
examples also put an activation after every Kernelized layer except the head.

**Two architecture variants (decided).** `act` is one setting (`train.activation` in `params.yaml`) that applies
to every `act` in the diagram:
- `tanh`: run first, through the whole experiment grid (Section 7).
- `relu`: run second, through the same grid. This is the Akida-compatible variant.

Applying ReLU to all four activation positions, including the projection and the head's hidden layer, is my
call: Akida needs ReLU everywhere, not only after the Kernelized layers.

### 4.1 `Kernelized` (faithful port of `akida_models.custom_layers.Kernelized`, v1.14.2)

Constructor: `Kernelized(in_channels, out_channels, length, num_coeffs=32, repeat=2, speed=4, speed_range=8)`.
`length` is the kernel length L, needed at construction because MetaTF initializes `log_dt` from the input
length at build time. Hidden state size `N = num_coeffs × repeat` (64 by default, as in MetaTF's KWS and UORED
models).

Parameters (all trainable; keep MetaTF's names and layouts so weights can later move to Keras):

| Name | Shape | Initial value |
|---|---|---|
| `A` | (N, 2) | rows `[log 0.5, π·n]` for n = 0…num_coeffs−1, tiled `repeat` times (`np.tile`) |
| `B` | (in, N) | all `1/√in` |
| `C` | (N, out) | `randn · √(2/N)` |
| `log_dt` | (N,) | `linspace(log(1/(L−1)) + log speed, log(1/(L−1)) + log speed + log speed_range, repeat)`, each value repeated `num_coeffs` times consecutively (`np.repeat`) |

There is **no D term** (matches MetaTF).

Discretization (`zoh_discretize`, simplified ZOH): `A_c = −exp(A[:,0]) + i·A[:,1]`, `dt = exp(log_dt)`,
`dtA = dt·A_c`, `B̂ = dt·B` (B̂ is real, scaled per state column), `Â = exp(dtA)`.

Forward on input `u` of shape [B, T, in] (any T; the kernel has T taps):
`y[b,t,d] = Σ_{j=0..t} Σ_c Σ_n u[b,t−j,c] · B̂[c,n] · Re(exp(j·dtA_n)) · C[n,d]`.
MetaTF computes this with a zero-padded FFT (length 2T, keep the first T outputs), via one of two equivalent
contraction orders chosen by a batch-size heuristic. The port may implement either order, or both. Only the
real part of the kernel is used. Precision: float32 / complex64, as in MetaTF.

### 4.2 `KernelizedLinear` (the "SSM-linear" layer)

A Kernelized layer fixed to `Â = 0`, `B̂ = I`, `D = 0`, with only `C` trainable. The state size equals the input
width. Its kernel is a single tap at lag 0, so `y[t] = u[t] @ C`: a bias-free linear map applied per tangent
step. `C` has shape (in, out).

Its MetaTF reference is `quantizeml.layers.StatefulProjection(units, use_bias=False)`, the layer
`convert_to_stateful` uses for projections: a Keras `Dense` with no bias, applied at every time step. The
testbench checks `KernelizedLinear` against it (Section 8). Initialization matches it: Glorot uniform,
`C ~ U(−a, a)` with `a = √(6 / (in + out))` (decided).

Note: "A = 0" in continuous time would mean `Â = exp(0) = 1`, an integrator (running sum). The decided
meaning is the discrete-time `Â = 0`.

### 4.3 Unchanged
`TangentSpace` (sub-window 60, step 20, OAS shrinkage, log map), and its causal alignment: the last
sub-window ends at the last raw sample of the training example.

## 5. Data

**Storage (decided: continuous recordings).** `build_dataset.py` saves each subject's preprocessed continuous
recording once (EMG [8, T] float32, angles [15, T]) instead of pre-cut training examples. Preprocessing is
unchanged: resample to 200 Hz, notch + bandpass, skip the first 0.25 s, per-subject z-score of the EMG.
File layout is the builder's call. This version is under 1 GB and does not depend on k or the stride, so
sweeping them never triggers a rebuild.

**Slicing.** `EMGFK(data_dir, subjects, length, stride)` cuts training examples at load time. The i-th
training example of a recording ends at raw sample `e = length − 1 + i·stride`; its input is
`emg[:, e−length+1 : e+1]` and its target is `angles[:, e]`. This matches today's alignment exactly: with k = 1
and stride 40 it reproduces the current `X.npy` / `y.npy` (regression test, Section 8).

**Fewer targets at the start.** A training example needs `length` raw samples before its target, so the first
target of each recording sits at raw sample `length − 1`. At k = 5 that drops about 20 targets per subject
(out of about 9,000), roughly 0.2% fewer training examples than today. Accepted.

**Kernel cap.** L must not exceed the number of tangent steps in the shortest recording:
`L_max = (T_min − 60) // 20 + 1`, where T_min is the shortest subject recording after preprocessing.
`L = min(k × 10, L_max)`. If capped, log a warning. Recordings are about 30 min long, so k = 1 and k = 5 are
nowhere near the cap.

**Evaluation stride (my call).** Validation and test always use stride 40, whatever the training stride.
- 40 is today's pipeline value, so evaluation stays as it is now.
- All training strides at the same k are scored on identical targets. If evaluation followed the training
  stride, a stride-1040 run would be scored on 26× fewer, different targets, and score differences would partly
  reflect which targets were scored.
- Across k, the k = 5 set lacks the first ~20 targets per subject (see above).
- The paper evaluated every 100 ms (stride 20). We keep 40 to leave the pipeline unchanged.

Validation and test examples use the same length as training (kernel mode).

## 6. Training (unchanged except as listed)

Unchanged: subject split (train 1–18, val 19, test 20), HuberLoss, AdamW (lr 2e-4, weight decay 1e-4),
cosine annealing over 15 epochs, batch size 256, gradient clip 1.0, checkpoint = best validation R².

Changes:
- `src/params.yaml` `train:` gains `k: 5`, `example_stride: 40` and `activation: tanh`, and `seed: 0` becomes
  `seeds: [0, 1, 2]`.
  SSM layer sizes stay as class defaults, following the README convention.
- `train.py` trains once per seed and writes per-seed outputs plus an aggregate `metrics.json`: per split, the
  mean and std over seeds of each metric, and the per-seed values.
- `evaluate()` adds **MAE in degrees** (mean |pred − y| over joints and samples) and **NMSE** (`1 − R²`, the
  paper's metric) next to MSE and R².
- The builder must confirm the angle channels in the `.fif` files are in degrees. If not, convert before MAE.

**Hyperparameter note.** MetaTF trains Kernelized models very differently: AdamW at lr 5e-3 (UORED) to 1e-2
(KWS), cosine decay with 10% linear warmup, weight decay 0.05, applied to all parameters including `A` and
`log_dt`. We deliberately keep the current settings (lr 2e-4, wd 1e-4, no warmup) so only the architecture
changes. If the model misses the threshold, the first suspect is that `A` and `log_dt` barely move at lr 2e-4.
The first follow-up experiment would be MetaTF's optimizer settings.

## 7. Experiments and success criteria

**Grid.** k ∈ {1, 5} × example stride ∈ {40 (reference, today's value), 240 (stateful input size), training
example length / 2, training example length}, 3 seeds each. In raw samples:

| k | L (tangent steps) | Training example | Strides |
|---|---|---|---|
| 1 | 10 | 240 | 40, 120, 240 (stride = length and stride = 240 coincide) |
| 5 | 50 | 1040 | 40, 240, 520, 1040 |

7 configurations × 3 seeds = 21 training runs per activation variant: all 21 with `tanh` first, then all 21
with `relu` (42 in total). Sweep with DVC experiments
(`dvc exp run -S train.k=… -S train.example_stride=…`). How to queue them on the SkyPilot machine is the
builder's call.

**Comparison target: the paper** (Colot et al., `docs/continuous_paper.pdf`). Cross-subject TRR on EMG-FK:
**MAE 16.71° ± 3.97** (Table II; mean ± std over 20 leave-one-subject-out folds). NMSE is only shown as a
boxplot (Fig. 4, median about 0.98). We compare the 3-seed mean test MAE on **subject 20 only**. Caveats: the
paper's figure is a 20-subject average; it used 3 frequency bands at 500 Hz, while we use the single-band
simplified TRR at 200 Hz.

**Selection.** For each activation variant, the configuration reported against the paper is the one with the
best validation (subject 19) MAE, never the best test score. Picking the best of 7 configurations by test score
would use the test subject to make a choice, and the reported number would be the luckiest of 7 noisy
estimates, which is optimistic. With a single test subject that bias is large. Validation selection keeps the
test score an honest estimate. Test scores of all configurations are still reported for transparency.

**Criteria (decided).** Measured on the selected configuration's 3-seed mean test MAE (subject 20):
- **Success:** below 16.71°.
- **Threshold:** within 0.5° of it (≤ 17.21°).

## 8. Testing (test-first)

Write each test first, see it fail, then implement. For `Kernelized` and `KernelizedLinear` this is a testbench
that runs against both implementations. Expected values come only from MetaTF, never from our own code. The
testbench must pass against the real MetaTF layers (`Kernelized`, `StatefulProjection`) before any PyTorch code
exists, and fail against PyTorch until it is built. Before the port, a runnable stand-in with the right
interface but naive behaviour (zero output, PyTorch default init) must fail on assertions, not just on a
missing import.

**Reference fixtures from real MetaTF.** A separate, throwaway Python environment (not the project env) with
`akida_models==1.14.2`, `quantizeml~=1.2.5`, `tensorflow` / `tf_keras ~=2.19`. A script
`tests/fixtures/make_kernelized_fixtures.py`, run only in that environment, writes `.npz` fixtures that are
committed to the repo. CI and the training env never import TensorFlow. Fixtures cover:
- `zoh_discretize` outputs for given A, B, log_dt.
- `Kernelized` forward outputs for several (in, out, num_coeffs, repeat, L, batch) settings, including L = 10
  and L = 50, with `force_full_conv` both on and off so both contraction orders are covered. Each fixture
  stores the weights, input and output.
- Gradients of a scalar loss with respect to the input and to A, B, C, log_dt.
- Initial A, B, log_dt for a given (in, L, num_coeffs, repeat, speed, speed_range). C is random in MetaTF, so
  only its shape and scale are checked.
- `StatefulProjection(use_bias=False)` forward outputs and gradients (input and kernel) for several (in, out)
  settings, on 3-D [B, T, in] and 2-D [B, in] inputs.

Tolerance: float32 (e.g. rtol 1e-4, atol 1e-5). For the `Kernelized` forward outputs and gradients, MetaTF's own
float32/complex64 rounding can exceed that atol (large kernel phase arguments), so those tolerances are sized by it:
the fixture script also computes a float64 exact reference (direct convolution, not FFT) and records MetaTF's
error `e` against it, refusing any config where `e / max|exact| >= 1e-4` (this ties the reference to MetaTF).
PyTorch must then be within `max(original, 2e)` of MetaTF (triangle inequality) and within `max(original, e)` of
the exact reference (no less accurate than MetaTF). The float64 reference only sizes the tolerance; MetaTF stays
the reference PyTorch is matched to.

**Testbench (runs on MetaTF and on PyTorch):**
- `Kernelized` matches every fixture (outputs, gradients, deterministic init, C scale).
- `Kernelized` is causal: changing input after step t leaves outputs ≤ t unchanged.
- `Kernelized` has exactly the parameters A, B, C, log_dt.
- `KernelizedLinear` matches the `StatefulProjection` fixtures (outputs, gradients), is memoryless, has a single
  parameter (C / Keras `kernel`), and its init is Glorot uniform (bounded by `a`, std ≈ `a/√3`).

**PyTorch-only tests:**
- `Model`: output shape [B, 15] for k = 1 and k = 5; no bias parameters anywhere.
- `EMGFK`: k = 1, stride 40 reproduces today's training examples and targets exactly (on a small synthetic
  recording); correct counts for other strides; kernel cap and its warning.
- `evaluate()`: MAE and NMSE on known inputs.

Add a `pytest` step to `.github/workflows/ci.yml` (CI currently only lints).

## 9. Files touched

| File | Change |
|---|---|
| `src/model/kernelized.py` (new) | `zoh_discretize`, `Kernelized`, `KernelizedLinear` |
| `src/model/ssm.py` (new), `src/model/gru.py` (deleted) | `SSMEncoder`, replacing `GRUEncoder` |
| `src/model/model.py` | Uses `SSMEncoder` and `KernelizedLinear`; takes the kernel length |
| `scripts/build_dataset.py` | Saves continuous recordings |
| `src/dataset.py` | `EMGFK` slices training examples by length and stride |
| `scripts/train.py` | Kernel length and cap, seed loop, eval stride 40, no bias init, aggregate metrics |
| `src/trainer.py` | MAE and NMSE in `evaluate()` |
| `src/params.yaml`, `dvc.yaml` | `k`, `example_stride`, `seeds`; DVC deps |
| `tests/` (new) | Fixtures, fixture script, tests |
| `.github/workflows/ci.yml` | pytest step |
| `README.md` | Model description and structure |

## 10. Known risks, later work and future experiments

- **Tanh vs ReLU.** Both variants are trained (Section 4). Only the `relu` variant is a candidate for Akida:
  MetaTF's Akida models use ReLU, and Tanh support on Akida is unconfirmed.
- **No bias.** Without biases or BatchNorm, the model must learn the targets' mean offset through its
  weights. MetaTF supplies offsets through BatchNorm. If accuracy suffers, this is the second suspect after the
  learning rate.
- **Front end on the host.** `TangentSpace` (eigendecomposition, matrix log) cannot run on Akida; it stays on
  the host.
- **Stateful equivalence (later work).** When stateful inference is added, test that streaming in chunks of the stateful
  input size reproduces kernel-mode outputs. The parameter layout here (A, B, C, log_dt as in MetaTF) is chosen
  so weights can move to a Keras `Kernelized` model and through `convert_to_stateful`.

### 10.1 Future experiments

None of these are in scope for this round. Each one changes a single thing relative to the selected
configuration from Section 7, and is judged with the same 3-seed, validation-selected rule.

**Streaming and architecture**
- **Stateful input size of 64 raw samples instead of 240.** 64 raw samples hold only one full 60-sample
  sub-window, so a stateful front end would need to carry sub-windows across chunks.
- **Downsampling between Kernelized layers.** MetaTF's `Stride(s)` keeps every s-th time step (`x[:, s−1::s]`)
  after a Kernelized layer, and `convert_to_stateful` folds it into the output projection. In stateful mode on
  Akida, later layers then run s× less often. A total stride of 2 would match the target rate (one target per
  40 raw samples = 2 tangent steps). L must be divisible by the total stride so the last tangent step is kept.
  Deferred because in kernel mode the sequences are tiny (10–50 tangent steps), so it saves little compute and
  changes the architecture we're comparing.

**Optimization**
- **MetaTF's optimizer settings** (Section 6 hyperparameter note): AdamW at lr 5e-3, cosine decay with 10%
  linear warmup, weight decay 0.05. First follow-up if the model misses the threshold.

**Regularization (adding or swapping)**
- **Dropout rate.** Sweep 0, 0.1 (current), 0.2 and 0.3 at the current positions.
- **SpatialDropout1D instead of Dropout.** MetaTF's choice: it drops whole channels across all time steps
  instead of single values, which suits sequences where neighbouring steps are highly correlated.
- **BatchNorm after each Kernelized layer.** MetaTF's block pattern (Kernelized → BatchNorm → activation). It
  also gives the model a learned offset without a bias, and folds into the weights at Akida conversion. It
  changes the no-bias, no-BatchNorm decision, so it is its own experiment.
- **Weight decay strength.** Sweep 1e-4 (current), 1e-3, 1e-2 and 0.05 (MetaTF). Separately: exclude `A` and
  `log_dt` from weight decay, since decaying them pushes the SSM's poles and time steps toward fixed values
  rather than shrinking weights. Common practice for S4-style layers; MetaTF does not do this.
- **Input augmentation.** Random per-training-example gain on the EMG (MetaTF's UORED training uses random gain
  augmentation), and small additive Gaussian noise. Applied before `TangentSpace`, on training examples only.
- **Early stopping.** The paper uses early stopping with patience 20 instead of a fixed 15 epochs. Try more
  epochs with patience on validation MAE.

Akida note: Dropout, SpatialDropout1D, weight decay, augmentation and early stopping exist only during training
and don't affect deployment. BatchNorm folds into the weights. LayerNorm is deliberately not listed because it
has no folding path to Akida.
