# SSM Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TRR's GRU encoder and linear layers with a faithful PyTorch port of MetaTF's Kernelized SSM
layers, train them in kernel mode on longer training examples, and sweep k × example stride × activation
against the paper.

**Architecture:** `src/model/kernelized.py` ports `akida_models.custom_layers.Kernelized` (v1.14.2) and adds a
bias-free `KernelizedLinear`; both are proven against reference fixtures from the real MetaTF layers (`Kernelized`, `StatefulProjection`).
The dataset switches to continuous recordings that `EMGFK` slices at load time, so k and the example stride
change without rebuilding. `train.py` loops over seeds and aggregates metrics; a sweep script runs the grid as
DVC experiments.

**Tech Stack:** Python ≥ 3.10, PyTorch, NumPy, DVC, pytest, ruff. TensorFlow 2.19 + akida_models 1.14.2 only in
a separate throwaway environment for fixture generation.

**Spec:** `docs/superpowers/specs/2026-11-08-ssm-encoder-design.md` (read it first: Section 3 defines every
size term used below).

## Global Constraints

- The project environment and CI never import TensorFlow or MetaTF. Only `tests/fixtures/make_kernelized_fixtures.py` does, and it runs in the separate environment.
- No layer has a bias. No BatchNorm.
- Kernelized parameters keep MetaTF's names and shapes: `A` (N, 2), `B` (in, N), `C` (N, out), `log_dt` (N,). No D term.
- float32 / complex64, as in MetaTF.
- Kernel length `L = k × 10` tangent steps; training example = `60 + (L − 1) × 20` raw samples; validation and test always use example stride 40.
- Unchanged: `TangentSpace` behaviour, preprocessing, subject split (val 19, test 20), HuberLoss, AdamW lr 2e-4 / wd 1e-4, cosine over 15 epochs, batch 256, grad clip 1.0, checkpoint on best val R².
- `ruff check .` and `ruff format --check .` pass (line length 120).
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A training example whose tangent-step count differs from the model's kernel length L: `Model.forward` raises `ValueError` (Task 4, `test_model_rejects_wrong_length`).
2. A recording shorter than one training example contributes zero examples, with no negative indices or crash (Task 5, `test_short_recording_yields_no_examples`).
3. k large enough to exceed the shortest recording: L is capped at `L_max` and a warning is emitted (Task 5, `test_kernel_length_caps_and_warns`).
4. A training stride other than 40: validation and test datasets still use stride 40 (Task 7, `test_eval_sets_use_stride_40`).
5. An activation name other than `tanh` / `relu`: `ValueError`, not a silent default (Task 4, `test_unknown_activation_raises`).

---

## Section 1: Kernelized layers (Tasks 1.0–3)

**Principle.** Every expected value comes from MetaTF, never from our own code. The testbench is written once,
proven on the real MetaTF layers, and then judges the PyTorch layers without being changed. Tolerances are
fixed in Task 1.0 and are never loosened to make PyTorch pass.

### Task 1.0: Testbench for both layers (passes on MetaTF, fails on PyTorch)

The testbench runs against two **backends**:
- `metatf`: the real `akida_models.custom_layers.Kernelized` and `quantizeml.layers.StatefulProjection(use_bias=False)`.
- `torch`: `src.model.kernelized.Kernelized` and `KernelizedLinear`, which don't exist yet.

Done when every test passes on `metatf`, fails on a naive stand-in on its assertions, and fails on `torch`.

**Files:**
- Create: `tests/kernelized_backends.py` (backend adapters)
- Create: `tests/test_kernelized.py` (the testbench)
- Create: `tests/fixtures/make_kernelized_fixtures.py`
- Create (generated, committed): `tests/fixtures/kernelized_{small,long,model,repeat3}.npz`, `tests/fixtures/projection_{small,model,head2d}.npz`

**Interfaces** (`tests/kernelized_backends.py`; all inputs and outputs NumPy float32, complex64 for complex values;
TensorFlow and torch are imported only inside the adapters, so the file imports cleanly in either environment):
- `kernelized(backend, in_channels, out_channels, length, num_coeffs, repeat, speed, speed_range)` returns an adapter with:
  - `.weights() -> dict` (`A, B, C, log_dt`) and `.set_weights(A, B, C, log_dt)`;
  - `.param_names() -> set[str]` (Keras names stripped to the last path part, e.g. `"A"`);
  - `.forward(x)` and `.forward_full(x)`. `forward_full` uses `force_full_conv=True` on `metatf`; on `torch` it equals `forward`;
  - `.grads(x, g) -> dict` with keys `x, A, B, C, log_dt` (gradients of `sum(forward(x) * g)`).
- `zoh(backend, A, B, log_dt) -> (A_hat, B_hat, dtA)`.
- `projection(backend, in_channels, out_channels)` returns an adapter with `.weights() -> {"C"}`,
  `.set_weights(C)`, `.param_names()` (Keras `kernel` reported as `"C"`), `.forward(x)` for [B, T, in] or
  [B, in], and `.grads(x, g) -> {"x", "C"}`.
- `BACKENDS`: `["torch"]`, plus `"metatf"` if `akida_models` and `quantizeml` are importable, or if the
  environment variables `METATF_CUSTOM_LAYERS` / `METATF_STATEFUL_RECURRENT` point to the source files.
- Kernelized fixture keys: meta `in_channels, out_channels, num_coeffs, repeat, speed, speed_range, length, batch`;
  `init_A, init_B, init_log_dt` (right after build, seeded); `A, B, C, log_dt` (init plus `0.1 · N(0, 1)` on
  every parameter, so trained-looking values are covered); `x`, `g`; `y_state`, `y_full`;
  `grad_x, grad_A, grad_B, grad_C, grad_log_dt`; `zoh_A_hat, zoh_B_hat, zoh_dtA`; `source`.
  Configs (name: in, out, num_coeffs, repeat, speed, speed_range, length, batch):
  `small`: 6, 5, 4, 2, 4, 8, 10, 3 · `long`: 16, 8, 32, 2, 4, 8, 50, 2 · `model`: 256, 128, 32, 2, 4, 8, 50, 4 ·
  `repeat3`: 5, 7, 8, 3, 2, 4, 20, 2.
- Projection fixture keys: `in_channels, out_channels`, `C`, `x`, `g`, `y`, `grad_x`, `grad_C`, `source`.
  Configs (name: in, out, x shape): `small`: 6, 5, [3, 10, 6] · `model`: 36, 256, [4, 50, 36] · `head2d`: 64, 15, [4, 64].

- [ ] **Step 1: Create the MetaTF environment (outside the repo)**

```bash
python -m venv ~/.venvs/metatf
~/.venvs/metatf/Scripts/pip install "akida_models==1.14.2" "tensorflow~=2.19.0" pytest
```

If `akida_models` fails to install on Windows, install `"tensorflow~=2.19.0" "tf_keras~=2.19.0" scipy pytest`,
run `pip download akida_models==1.14.2 quantizeml==1.2.5 --no-deps` into a scratch dir, extract both, and set
`METATF_CUSTOM_LAYERS` to `akida_models/custom_layers.py`. For `StatefulProjection`, either set
`METATF_STATEFUL_RECURRENT` to `quantizeml/layers/stateful_recurrent.py` if it imports standalone, or install
`quantizeml==1.2.5` on its own (its dependencies are lighter than `akida_models`'s).

- [ ] **Step 2: Write `make_kernelized_fixtures.py`**

It uses the `metatf` adapters only. Kernelized, per config: seed numpy and TF with 0, build, record `init_*`,
perturb and `set_weights`, draw `x` and `g` from N(0, 1), then record forward, forward_full, grads and zoh.
Projection, per config: seed 0, build, keep its Glorot-initialized `C`, draw `x` and `g`, record forward and
grads. The module docstring gives the Step 1 commands and says the script is never imported by tests or CI.

- [ ] **Step 3: Write the testbench `tests/test_kernelized.py`**

Every test is parametrized over `BACKENDS` and, where it uses fixtures, over the configs. It uses
`numpy.testing.assert_allclose` only (no torch import at module level).

```python
# Kernelized
def test_init_matches_metatf(backend, name):          # A, B, log_dt == init_* (rtol 1e-6, atol 1e-6); C shape == fixture C shape
def test_C_init_scale(backend):                        # kernelized(256, 128, 50, 32, 2, 4, 8): |std(C) / sqrt(2/64) - 1| < 0.05
def test_zoh_matches_metatf(backend, name):            # zoh(backend, A, B, log_dt) == zoh_* (rtol 1e-5, atol 1e-6)
def test_forward_matches_metatf(backend, name):        # set_weights(fixture); forward and forward_full vs y_state and y_full (rtol 1e-4, atol 1e-5)
def test_gradients_match_metatf(backend, name):        # grads(x, g) vs grad_* (rtol 1e-3, atol 1e-4)
def test_causal(backend):                              # kernelized(6, 5, 10, …); perturb x[:, 7:]; forward(x)[:, :7] unchanged (atol 1e-6)
def test_parameters_are_A_B_C_log_dt(backend):         # param_names() == {"A", "B", "C", "log_dt"}: no bias, no D

# KernelizedLinear vs StatefulProjection(use_bias=False)
def test_linear_forward_matches_metatf(backend, name): # set_weights(C); forward(x) vs y (rtol 1e-5, atol 1e-6)
def test_linear_gradients_match_metatf(backend, name): # grads(x, g) vs grad_x, grad_C (rtol 1e-4, atol 1e-5)
def test_linear_memoryless(backend):                   # projection(6, 5); perturb x[:, 3] only; outputs differ at step 3 and nowhere else
def test_linear_only_C(backend):                       # param_names() == {"C"}: no bias
def test_linear_init_glorot(backend):                  # projection(256, 512): a = sqrt(6/768); all |C| <= a; |std(C) / (a/sqrt(3)) - 1| < 0.05
```

- [ ] **Step 4: Generate the fixtures**

Run: `~/.venvs/metatf/Scripts/python tests/fixtures/make_kernelized_fixtures.py`
Expected: seven `.npz` files, each under 1 MB.

- [ ] **Step 5: Confirm the testbench passes on MetaTF**

Run: `~/.venvs/metatf/Scripts/python -m pytest tests/test_kernelized.py -k metatf -v`
Expected: every `metatf` test PASSES. (`torch` tests fail in this env because torch isn't installed; ignore them.)

- [ ] **Step 6: Confirm the testbench catches a naive stand-in**

Temporarily create `src/model/kernelized.py` with `Kernelized` and `KernelizedLinear` that have the Task 2/3
signatures and parameter names but naive behaviour: PyTorch default init (`torch.randn` for every parameter),
`forward` returns zeros of the right shape, and `zoh_discretize` returns its inputs unchanged.
Run (project env): `pytest tests/test_kernelized.py -v`
Expected: every fixture-comparison and init test FAILS on an assertion (not on an import error). Record which
tests, if any, pass on the stand-in. Only the structural tests (`test_parameters_are_A_B_C_log_dt`,
`test_linear_only_C`) may pass. Then **delete the stand-in file**; it is never committed.

- [ ] **Step 7: Confirm the testbench fails on PyTorch**

Run (project env): `pytest tests/test_kernelized.py -v`
Expected: every collected test is `torch`, and each FAILS with `ModuleNotFoundError: No module named 'src.model.kernelized'`.

- [ ] **Step 8: Commit**

```bash
git add tests/
git commit -m "test: Kernelized/StatefulProjection testbench validated on MetaTF, failing on PyTorch"
```

---

### Task 2: PyTorch `zoh_discretize` and `Kernelized`

**Files:**
- Create: `src/model/kernelized.py`
- Modify: `tests/kernelized_backends.py` only if the `torch` adapter needs a fix to match the interface below. Never change expected values or tolerances.

**Interfaces:**
- Consumes: the Task 1.0 testbench and fixtures.
- Produces:
  - `zoh_discretize(A: Tensor, B: Tensor, log_dt: Tensor) -> tuple[Tensor, Tensor, Tensor]`: returns `(A_hat, B_hat, dtA)`. `A_hat` and `dtA` are complex64 [N]; `B_hat` is float32 [in, N].
  - `Kernelized(in_channels: int, out_channels: int, length: int, num_coeffs: int = 32, repeat: int = 2, speed: float = 4, speed_range: float = 8)`, an `nn.Module` with parameters `A, B, C, log_dt`. `forward(u: Tensor[B, T, in]) -> Tensor[B, T, out]`, for any T.

- [ ] **Step 1: Run the Kernelized tests and confirm they fail**

Run: `pytest tests/test_kernelized.py -k "not linear" -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 2: Implement `src/model/kernelized.py`**

Initialization and discretization exactly as spec §4.1, including `np.tile` for `A`/`B` and `np.repeat` for
`log_dt`. For forward, use MetaTF's "internal states" order: `einsum('blc,cn->bnl', u, B_hat)`, causal FFT
convolution with `K = Re(exp(l · dtA))` for l = 0…T−1 (rfft/irfft with n = 2T, keep the first T), then
`einsum('bnl,nd->bld', ·, C)`. Both MetaTF paths are mathematically identical, so one path suffices; the
testbench checks against both outputs.

- [ ] **Step 3: Run the Kernelized tests and confirm they pass**

Run: `pytest tests/test_kernelized.py -k "not linear" -v && ruff check . && ruff format --check .`
Expected: all PASS; ruff clean. (The `linear` tests still fail until Task 3.) If a comparison fails only by float
noise, stop and report the max error instead of changing the tolerance. (Superseded for the `Kernelized` forward and
gradient tests by Task 1.1 / ledger Ruling 6: those tolerances are sized by MetaTF's measured float32 rounding error.)

- [ ] **Step 4: Commit**

```bash
git add src/model/kernelized.py
git commit -m "feat: PyTorch port of MetaTF Kernelized SSM layer"
```

---

### Task 3: PyTorch `KernelizedLinear`, and CI

**Files:**
- Modify: `src/model/kernelized.py`
- Modify: `.github/workflows/ci.yml` (add a `Test` step running `pytest` after `Lint`)
- Modify: `pyproject.toml` (add `[tool.pytest.ini_options]` with `testpaths = ["tests"]`)

**Interfaces:**
- Consumes: the Task 1.0 testbench (`linear` tests, checked against `StatefulProjection`).
- Produces: `KernelizedLinear(in_channels: int, out_channels: int)`, an `nn.Module` with a single parameter
  `C` [in, out], Glorot-uniform initialized (`U(−a, a)`, `a = √(6 / (in + out))`). `forward(u: Tensor[..., in]) -> Tensor[..., out]`
  returns `u @ C`. Its docstring states Â = 0, B̂ = I, D = 0, state size = input width, and that the MetaTF
  reference is `StatefulProjection(use_bias=False)`.

- [ ] **Step 1: Run the linear tests and confirm they fail**

Run: `pytest tests/test_kernelized.py -k linear -v` → FAIL (`ImportError: cannot import name 'KernelizedLinear'`).

- [ ] **Step 2: Implement `KernelizedLinear`**

- [ ] **Step 3: Run the whole testbench and confirm it passes**

Run: `pytest -v && ruff check . && ruff format --check .` → all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/model/kernelized.py .github/workflows/ci.yml pyproject.toml
git commit -m "feat: KernelizedLinear (A_hat=0, B_hat=I, D=0) checked against StatefulProjection; pytest in CI"
```

---

## Section 2: Model, data and training pipeline (Tasks 4–8)

### Task 4: `SSMEncoder` and `Model`

**Files:**
- Create: `src/model/ssm.py`
- Delete: `src/model/gru.py`
- Modify: `src/model/model.py`
- Modify: `src/model/tangent_space.py` (add module-level helpers only; `TangentSpace` behaviour unchanged)
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: `Kernelized`, `KernelizedLinear` (Tasks 2–3).
- Produces:
  - `src/model/tangent_space.py`: `SUB_WINDOW = 60`, `STEP = 20` (used as `TangentSpace` defaults); `n_steps(raw_samples: int) -> int` = `(raw_samples − 60) // 20 + 1` (0 if fewer than 60); `raw_length(steps: int) -> int` = `60 + (steps − 1) · 20`.
  - `src/model/ssm.py`: `activation(name: str) -> nn.Module` (`"tanh"` → `nn.Tanh()`, `"relu"` → `nn.ReLU()`, otherwise `ValueError`); `SSMEncoder(in_dim: int, length: int, activation: str, dense: int = 256, units=(256, 128), dropout=0.1)` with `out_dim = 128` and `forward(x: [B, L, in_dim]) -> [B, 128]`, laid out per spec §4.
  - `src/model/model.py`: `Model(channels: int, targets: int, length: int, activation: str = "tanh", hidden: int = 64)` with attributes `tangent_space`, `encoder`, `head`, `length`. `forward(x: [B, C, raw_length(length)]) -> [B, targets]`; raises `ValueError` if the tangent-step count ≠ `length`.

- [ ] **Step 1: Write the failing tests**

```python
def test_step_helpers():
    assert (n_steps(240), n_steps(1040), n_steps(64), n_steps(59)) == (10, 50, 1, 0)
    assert (raw_length(10), raw_length(50)) == (240, 1040)


@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("act", ["tanh", "relu"])
def test_model_output_shape(k, act):
    model = Model(8, 15, 10 * k, act)
    assert model(torch.randn(4, 8, raw_length(10 * k))).shape == (4, 15)


def test_model_has_no_bias():
    names = [n for n, _ in Model(8, 15, 10).named_parameters()]
    assert not any("bias" in n for n in names)


def test_activation_variant_is_used_everywhere():
    for act, cls in [("tanh", nn.Tanh), ("relu", nn.ReLU)]:
        mods = list(Model(8, 15, 10, act).modules())
        assert sum(isinstance(m, cls) for m in mods) == 4
        assert not any(isinstance(m, nn.Tanh if cls is nn.ReLU else nn.ReLU) for m in mods)


def test_unknown_activation_raises():
    with pytest.raises(ValueError):
        Model(8, 15, 10, "gelu")


def test_model_rejects_wrong_length():
    with pytest.raises(ValueError):
        Model(8, 15, 50)(torch.randn(2, 8, 240))


def test_layer_widths():
    m = Model(8, 15, 10)
    assert (m.encoder.proj.C.shape, m.encoder.ssm1.C.shape, m.encoder.ssm2.C.shape) == ((36, 256), (64, 256), (64, 128))
    assert (m.head[0].C.shape, m.head[2].C.shape) == ((128, 64), (64, 15))
```

The attribute names `proj`, `ssm1`, `ssm2`, `head[0]`, `head[2]` are part of this interface.

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_model.py -v` → FAIL (`ImportError` for `n_steps` / `src.model.ssm`).

- [ ] **Step 3: Implement**

`SSMEncoder.forward`: `h = act(proj(x)); h = dropout(act(ssm1(h))); h = act(ssm2(h)); return dropout(h[:, -1])`.
Each `act` position is a separate module instance. `Model.head = nn.Sequential(KernelizedLinear(128, hidden), activation(act), KernelizedLinear(hidden, targets))`.
Update the `Model` docstring to describe the SSM variant. Delete `gru.py`.

- [ ] **Step 4: Run and confirm they pass**

Run: `pytest -v && ruff check . && ruff format --check .` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src/model tests/test_model.py
git commit -m "feat: SSM encoder replaces GRU encoder, tanh/relu variants"
```

---

### Task 5: Continuous-recording dataset

**Files:**
- Modify: `src/dataset.py`
- Modify: `scripts/build_dataset.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `n_steps` (Task 4).
- Produces (in `src/dataset.py`):
  - `save_recording(path: Path, emg: np.ndarray, angles: np.ndarray) -> None` and `load_recording(path) -> tuple[np.ndarray, np.ndarray]`. Format: `subject_{s:02d}.npz` with `emg` [8, T] and `angles` [15, T], both float32.
  - `recording_path(data_dir, subject: int) -> Path`.
  - `shortest_recording(data_dir) -> int`: the minimum T, in raw samples, over all `subject_*.npz`.
  - `kernel_length(k: int, shortest: int, stateful_input: int = 240) -> int` = `min(k · n_steps(stateful_input), n_steps(shortest))`, issuing `warnings.warn` when capped.
  - `EMGFK(data_dir, subjects, length: int, stride: int)`, a `Dataset` with attributes `channels` (8) and `targets` (15). Item i is `(emg[:, e−length+1 : e+1], angles[:, e])` as float32 tensors, with `e = length − 1 + j · stride` within its subject's recording.

- [ ] **Step 1: Write the failing tests** (synthetic recordings written to `tmp_path` with `save_recording`; subject s has T_s raw samples and `emg = rng.standard_normal((8, T_s))`)

```python
def test_k1_stride40_matches_old_windows(tmp_path):    # the pre-change build logic
    emg, angles = make(tmp_path, 1, T=2000)
    ds = EMGFK(tmp_path, [1], 240, 40)
    old_X = sliding_window_view(emg, 240, axis=1)[:, ::40].transpose(1, 0, 2)
    old_y = angles[:, 239::40].T
    assert len(ds) == len(old_X)
    for i in range(len(ds)):
        assert np.array_equal(ds[i][0].numpy(), old_X[i]) and np.array_equal(ds[i][1].numpy(), old_y[i])

@pytest.mark.parametrize("length,stride", [(1040, 40), (1040, 520), (1040, 1040), (240, 120)])
def test_counts_and_alignment(tmp_path, length, stride):
    emg, angles = make(tmp_path, 1, T=5000)
    ds = EMGFK(tmp_path, [1], length, stride)
    assert len(ds) == (5000 - length) // stride + 1
    e = length - 1 + 3 * stride
    assert np.array_equal(ds[3][0].numpy(), emg[:, e - length + 1 : e + 1]) and np.array_equal(ds[3][1].numpy(), angles[:, e])

def test_subject_filter_and_concat(tmp_path):        # subjects 1 (T=3000), 2 (T=4000), 3; EMGFK(.., [1, 2], ..) counts both, excludes 3

def test_short_recording_yields_no_examples(tmp_path):
    make(tmp_path, 1, T=500); make(tmp_path, 2, T=3000)
    ds = EMGFK(tmp_path, [1, 2], 1040, 40)
    assert len(ds) == (3000 - 1040) // 40 + 1

def test_shortest_recording(tmp_path):  # T = 3000, 4000 -> 3000

def test_kernel_length_caps_and_warns():
    assert kernel_length(5, 360_000) == 50
    with pytest.warns(UserWarning): assert kernel_length(5, 500) == n_steps(500)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_dataset.py -v` → FAIL (`ImportError: save_recording`).

- [ ] **Step 3: Implement `src/dataset.py`**

Load every selected recording into memory (under 1 GB total). Build a flat index of `(subject_position, end)`
pairs in `__init__`.

- [ ] **Step 4: Update `scripts/build_dataset.py`**

Keep download and preprocessing exactly as they are, up to and including the per-subject z-score. Replace the
`sliding_window_view` / concatenate / `np.save` part with `save_recording(recording_path(out_dir, s), emg, angles)`
and print `T` per subject. Remove the `WINDOW` and `STRIDE` constants.

- [ ] **Step 5: Verify angle units on real data**

Run `python scripts/build_dataset.py` (downloads about 2 GB on first run). Then print, for `Subject_01.fif`,
`raw.info["chs"][i]["unit"]` for the angle channels and the min/max of `angles`. Expected: values within
roughly ±180, i.e. degrees. If they are radians (|values| ≤ ~3.2), convert with `np.degrees` before saving, and
add that to the commit message. Record the observed unit in the PR description.

- [ ] **Step 6: Run and confirm everything passes**

Run: `pytest -v && ruff check . && ruff format --check .` → all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/dataset.py scripts/build_dataset.py tests/test_dataset.py
git commit -m "feat: store continuous recordings, slice training examples by length and stride"
```

---

### Task 6: MAE and NMSE metrics

**Files:**
- Modify: `src/trainer.py` (`evaluate`)
- Create: `tests/test_trainer.py`

**Interfaces:**
- Produces: `evaluate(model, loader, device) -> dict` with keys `mse`, `r2` (unchanged), `nmse` (= mean over joints of mse/var, i.e. `1 − r2`) and `mae` (mean |pred − y| over joints and samples, in the targets' units: degrees).

- [ ] **Step 1: Write the failing test**

```python
def test_evaluate_metrics():
    y = torch.tensor([[0.0, 10.0], [2.0, 14.0]])
    pred = torch.tensor([[1.0, 10.0], [2.0, 10.0]])
    loader = [(torch.zeros(2, 1), y)]
    m = evaluate(ConstModel(pred), loader, "cpu")  # ConstModel returns pred regardless of input
    assert m["mae"] == pytest.approx((1 + 0 + 0 + 4) / 4)
    assert m["nmse"] == pytest.approx(1 - m["r2"])
    assert m["nmse"] == pytest.approx(((0.5 / 1.0) + (8.0 / 4.0)) / 2)
```

- [ ] **Step 2: Run it and confirm it fails** (`KeyError: 'mae'`)

- [ ] **Step 3: Implement**

- [ ] **Step 4: Run and confirm it passes**: `pytest tests/test_trainer.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/trainer.py tests/test_trainer.py
git commit -m "feat: report MAE (degrees) and NMSE"
```

---

### Task 7: Training script: seeds, kernel length, eval stride, aggregate metrics

**Files:**
- Modify: `scripts/train.py`
- Modify: `src/params.yaml`
- Modify: `README.md`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: `EMGFK`, `shortest_recording`, `kernel_length` (Task 5); `raw_length` (Task 4); `Model` (Task 4); `evaluate`, `fit`, `save_history` (Task 6 / existing).
- Produces in `scripts/train.py`:
  - `EVAL_STRIDE = 40`.
  - `load_data(params, data_dir, length) -> (train_set, train_loader, val_loader, test_loader)`: train uses `params["example_stride"]`; val and test use `EVAL_STRIDE`.
  - `run(params: dict, data_dir: Path, out_dir: Path) -> dict`: for each seed in `params["seeds"]`, calls `torch.manual_seed(seed)`, builds `Model(train_set.channels, train_set.targets, L, params["activation"])` with no bias init, runs `fit` with checkpoint `out_dir/seed_{seed}/best_model.pt`, and tests the best checkpoint. Writes `out_dir/history.csv` (all seeds, with a `seed` column) and `out_dir/metrics.json`, and returns the metrics dict:
    `{"kernel_length": L, "example_length": raw_length(L), "val": {metric: {"mean", "std"}}, "test": {…}, "seeds": {"0": {"val": {…}, "test": {…}}, …}}`. std is the sample std (ddof = 1).
  - `__main__`: reads `src/params.yaml["train"]` and calls `run(params, DATA_DIR, OUT_DIR)`.
- `src/params.yaml` `train:` adds `k: 5`, `example_stride: 40`, `activation: tanh` and replaces `seed: 0` with `seeds: [0, 1, 2]`.

- [ ] **Step 1: Write the failing tests** (synthetic recordings for subjects 1–4, T = 3000 each; params with `val_subjects: [3]`, `test_subjects: [4]`, `epochs: 1`, `batch_size: 8`, `k: 1`, `seeds: [0, 1]`, `device: cpu`; the other keys copied from `src/params.yaml`)

```python
def test_run_writes_aggregate_metrics(tmp_path):
    m = run(params, data_dir, tmp_path / "out")
    assert m["kernel_length"] == 10 and m["example_length"] == 240
    assert set(m["seeds"]) == {"0", "1"}
    for split in ("val", "test"):
        assert set(m[split]) == {"mse", "r2", "nmse", "mae"}
        assert m[split]["mae"]["mean"] == pytest.approx(np.mean([m["seeds"][s][split]["mae"] for s in ("0", "1")]))
    assert json.loads((tmp_path / "out/metrics.json").read_text()) == m
    assert (tmp_path / "out/seed_0/best_model.pt").exists()


def test_eval_sets_use_stride_40(tmp_path):
    _, _, val_loader, test_loader = load_data({**params, "example_stride": 1040}, data_dir, 240)
    assert len(val_loader.dataset) == (3000 - 240) // 40 + 1
    assert len(test_loader.dataset) == (3000 - 240) // 40 + 1
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_train.py -v` → FAIL (`ImportError: cannot import name 'run'`). Import with `from scripts.train import run, load_data`; add an empty `scripts/__init__.py` if needed.

- [ ] **Step 3: Implement `scripts/train.py` and update `src/params.yaml`**

Compute `L = kernel_length(params["k"], shortest_recording(data_dir))` once, before the seed loop.

- [ ] **Step 4: Update `README.md`**

Describe the SSM model, the continuous-recording dataset, the new `params.yaml` keys and the `output/` layout. In
the Structure block, replace `gru.py` with `kernelized.py` and `ssm.py`. Leave `dvc.yaml` as is: `src/model`, the
`train` params and `output/history.csv` are already declared.

- [ ] **Step 5: Run and confirm everything passes**

Run: `pytest -v && ruff check . && ruff format --check .` → all PASS.

- [ ] **Step 6: Real-data smoke run**

Run: `python scripts/train.py` with `epochs: 1` and `seeds: [0]` set temporarily (revert afterwards).
Expected: one epoch completes on GPU, `output/metrics.json` contains `kernel_length: 50` and `example_length: 1040`, and the val MAE is a finite number of degrees.

- [ ] **Step 7: Commit**

```bash
git add scripts/ src/params.yaml README.md tests/test_train.py
git commit -m "feat: seed loop, kernel-length cap, fixed eval stride, aggregate metrics"
```

---

### Task 8: Sweep tooling and selection rule

**Files:**
- Create: `scripts/sweep.sh`
- Create: `scripts/summarize_sweep.py`
- Create: `tests/test_summarize_sweep.py`
- Modify: `sky/train.yaml` (env `SWEEP: ""`; if set, run `bash scripts/sweep.sh "$SWEEP"` instead of `dvc repro`)
- Modify: `README.md` (how to run and fetch a sweep)

**Interfaces:**
- `scripts/sweep.sh ACTIVATION`: runs `dvc pull --allow-missing`, `dvc repro build_dataset` and `dvc push`, then one `dvc exp run -S train.activation=$1 -S train.k=$k -S train.example_stride=$s` per configuration: k=1 with strides 40, 120, 240; k=5 with strides 40, 240, 520, 1040. Finally it runs `dvc exp push` if a git remote exists and writes `dvc exp show --csv > sweep_$1.csv`.
- `scripts/summarize_sweep.py`:
  - `select(rows: list[dict]) -> dict` returns the row with the lowest `val.mae.mean` (never looks at test).
  - `verdict(test_mae: float) -> str` returns `"success"` if < 16.71, `"threshold"` if ≤ 17.21, else `"miss"`.
  - CLI `python scripts/summarize_sweep.py sweep_tanh.csv` prints every configuration's val/test MAE, the selected configuration and its verdict. Columns are matched by suffix (`val.mae.mean`, `test.mae.mean`, `train.k`, `train.example_stride`, `train.activation`).

- [ ] **Step 1: Write the failing tests**

```python
def test_select_uses_validation_not_test():
    rows = [{"val.mae.mean": 17.0, "test.mae.mean": 15.0}, {"val.mae.mean": 16.0, "test.mae.mean": 18.0}]
    assert select(rows) is rows[1]


@pytest.mark.parametrize(
    "mae,expected", [(16.70, "success"), (16.71, "threshold"), (17.21, "threshold"), (17.22, "miss")]
)
def test_verdict(mae, expected):
    assert verdict(mae) == expected
```

- [ ] **Step 2: Run them and confirm they fail**, then **implement**, then **run them and confirm they pass** (`pytest -v`, ruff clean).

- [ ] **Step 3: Commit**

```bash
git add scripts/sweep.sh scripts/summarize_sweep.py tests/test_summarize_sweep.py sky/train.yaml README.md
git commit -m "feat: k x stride sweep via DVC experiments and validation-based selection"
```

---

## Section 3: Experiments

### Task 9: Run the experiments (needs the user's go-ahead: GPU cost)

- [ ] **Step 1: Ask the user to confirm launching 21 GPU runs on SkyPilot for `tanh`.** Do not launch without a yes.
- [ ] **Step 2: Run the tanh grid.** `sky exec hri-gpu sky/train.yaml --env SWEEP=tanh`, then `rsync hri-gpu:sky_workdir/sweep_tanh.csv .`, then `python scripts/summarize_sweep.py sweep_tanh.csv`.
- [ ] **Step 3: Report to the user:** the table, the selected configuration and its verdict against 16.71° / 17.21°.
- [ ] **Step 4: Repeat Steps 1–3 for `relu`** once the user confirms.
