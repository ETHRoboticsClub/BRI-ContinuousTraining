# Extra Experiments (Task 13) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the ten opt-in switches of Task 13 (three augmentations, one normalization, six SSM
regularizations), each defaulting to today's behaviour, plus a sweep that runs each one alone against a
baseline rerun.

**Architecture:** Normalization moves from `build_dataset.py` to load time in `EMGFK`, so per-subject and
pooled z-scoring can be compared without a second dataset. Augmentation is a seeded callable that `EMGFK`
applies to training examples in `__getitem__`. Downsampling and `speed` are `SSMEncoder` arguments; the SSM
learning-rate and weight-decay changes are AdamW parameter groups. A new sweep script names each DVC experiment
after its change, and `summarize_sweep.py` builds the Δ-versus-baseline table from those names.

**Tech Stack:** Python ≥ 3.10, PyTorch, NumPy, DVC, pytest, ruff, bash.

**Spec:** `docs/superpowers/plans/2026-11-08-ssm-encoder.md`, Section 5 / Task 13 (the change table and "My
calls"), and `docs/superpowers/specs/2026-11-08-ssm-encoder-design.md` (§3 terms, §5 data, §10.1). This plan
replaces Task 13's Files and Steps. Its change table and decisions stand.

**Mode.** Everything here is kernel-mode training: the model sees training examples of L tangent steps
(L = 10 at k = 1, 50 at k = 5). Stateful streaming stays out of scope. Augmentation, normalization statistics
and parameter groups exist only in training. `downsample` and `ssm_speed` change the architecture or its
initialization, so a later stateful or Akida conversion has to carry them over.

**Prerequisites.** Tasks 1–8 are merged. Tasks 10–11 (Section 4) are not required. If they have merged when
this plan runs, keep their keyword arguments and add this plan's after them. In Task 13.8, reuse Task 11's
Δ-table printer instead of writing a second one.

## Global Constraints

- **Defaults are the baseline.** With the default `src/params.yaml`, the training examples, model and
  optimizer are those of today, within float32 rounding.
- **One change per experiment.** Each experiment differs from its baseline rerun in exactly one switch.
- Validation and test examples are never augmented and always use example stride 40.
- Every switch value outside its domain raises `ValueError` (no silent default).
- New source files go into `dvc.yaml`'s `train` deps.
- Unchanged from the main plan: no TensorFlow import, float32 / complex64, no bias (BatchNorm only via Section
  4's opt-in switch), `ruff check .` and `ruff format --check .` pass (line length 120).
- Terms as in spec §3. "Window" alone is never used.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

**New `src/params.yaml` `train:` keys** (defaults = baseline; each task adds its own):

```yaml
  normalization: subject  # none, subject or set
  aug_gain: 0.0           # std of log per-channel gain; 0 = off
  aug_rotate: 0           # max whole-electrode ring rotation; 0 = off
  aug_shift: 0.0          # max fractional ring shift in electrodes; 0 = off
  downsample: 1           # keep every s-th tangent step after ssm1
  ssm_speed: 4            # Kernelized `speed` of ssm1 and ssm2
  ssm_lr_scale: 1.0       # LR multiplier for A and log_dt
  ssm_decay_A: true       # false: no weight decay on A
```

## Review Focus

1. Default `normalization` on the new un-normalized recordings: training examples must equal today's z-scored
   ones, in tests and on real data (Task 13.1, `test_subject_normalization_matches_old_build`, Step 6).
2. Augmentation reaching validation or test examples, or drawing from the global torch RNG and so shifting the
   baseline's shuffle order and dropout masks (Task 13.2, `test_only_training_examples_are_augmented`,
   `test_augment_leaves_global_rng_alone`).
3. A `downsample` that leaves `ssm2` with one tangent step (L / s = 1 gives `log(1/0)` in `Kernelized`
   init) or does not divide L: `ValueError`, not `ZeroDivisionError` or a dropped last step (Task 13.5,
   `test_bad_downsample_raises`).
4. Parameter groups that miss or duplicate a parameter, or a scheduler that loses the 0.1 LR ratio (Task 13.7,
   `test_param_groups_cover_every_parameter_once`, `test_ssm_lr_ratio_survives_cosine`).
5. `BASE_SETS` setting a key that one of the changes also sets, so an experiment changes nothing or two
   things: the sweep refuses to start (Task 13.8, `test_sweep_extra_rejects_conflicting_base`).

---

### Task 13.1: Load-time normalization and N1 (`normalization: set`)

`build_dataset.py` stops z-scoring. `EMGFK` z-scores at load time: per recording (`subject`, today's
behaviour), pooled over the dataset's own recordings (`set`), or not at all (`none`, used by the slicing
tests). Validation and test are one subject each, so `set` equals `subject` there and N1 only changes
training.

**Files:**
- Modify: `scripts/build_dataset.py` (delete the z-score line; save the filtered EMG)
- Modify: `src/dataset.py`, `scripts/train.py`, `src/params.yaml`, `README.md`
- Modify: `tests/test_dataset.py`, `tests/test_train.py`
- Modify (generated): `dvc.lock` after the rebuild

**Interfaces:**
- Produces in `src/dataset.py`:
  - `NORMALIZATIONS = ("none", "subject", "set")`, `EPS = 1e-6`.
  - `channel_stats(emgs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]`: per-channel mean and std,
    each [8, 1], over the recordings concatenated along time. Uses `x.mean(1, keepdims=True)` and
    `x.std(1, keepdims=True)`, today's ops, so one recording reproduces today's statistics exactly.
  - `EMGFK(data_dir, subjects, length, stride, normalization="subject", augment=None)`. It stores each
    recording as `(emg − mean) / (std + EPS)`, with stats from `channel_stats([emg])` for `subject` and from
    `channel_stats(all of this dataset's recordings)` for `set`. Other values raise `ValueError`. (`augment`
    is used from Task 13.2 on.)
- `scripts/train.py` `load_data` passes `params["normalization"]` to all three datasets.

- [ ] **Step 1: Write the failing tests**

In `tests/test_dataset.py`, the existing slicing tests (`test_k1_stride40_matches_old_windows`,
`test_counts_and_alignment`, `test_subject_filter_and_concat`) get `normalization="none"`. Their expected
values stay as they are. New tests:

```python
def test_subject_normalization_matches_old_build(tmp_path):
    rng = np.random.default_rng(0)
    raw = (3e-5 * rng.standard_normal((8, 3000)) + rng.standard_normal((8, 1)) * 1e-6).astype(np.float32)
    save_recording(recording_path(tmp_path, 1), raw, np.zeros((15, 3000)))
    old = (raw - raw.mean(1, keepdims=True)) / (raw.std(1, keepdims=True) + 1e-6)  # today's build_dataset line
    ds = EMGFK(tmp_path, [1], 240, 40)  # default normalization
    np.testing.assert_allclose(ds[5][0].numpy(), old[:, 200:440], rtol=1e-6, atol=1e-7)  # e = 239 + 5 * 40


def test_set_normalization_pools_the_datasets_subjects(tmp_path):
    e1, e2 = scaled recordings of subjects 1 and 2 (stds 1e-5 and 4e-5), plus subject 3 (not loaded)
    ds = EMGFK(tmp_path, [1, 2], 240, 40, normalization="set")
    cat = np.concatenate([e1, e2], 1)
    expected = (e2 - cat.mean(1, keepdims=True)) / (cat.std(1, keepdims=True) + 1e-6)
    # first example of subject 2 == expected[:, :240] (rtol 1e-5, atol 1e-6); subject 3 does not enter the stats


def test_set_equals_subject_for_one_subject(tmp_path):    # same example, normalization "set" vs "subject": equal
def test_none_returns_stored_emg(tmp_path):               # normalization="none": ds[0][0] == stored emg[:, :240]
def test_unknown_normalization_raises(tmp_path):          # normalization="global" -> ValueError
```

In `tests/test_train.py`:

```python
def test_set_normalization_changes_training_only(data_dir):
    sets = {n: load_data({**PARAMS, "normalization": n}, data_dir, 240) for n in ("subject", "set")}
    for loader in (2, 3):  # val, test: one subject each, so pooled == per-subject
        assert torch.equal(sets["subject"][loader].dataset[0][0], sets["set"][loader].dataset[0][0])
    assert not torch.equal(sets["subject"][0][0][0], sets["set"][0][0][0])
```

(The `data_dir` fixture's subjects need different scales for the last assertion; scale subject s by `s`.)

- [ ] **Step 2: Run them and confirm they fail**

Run: `pytest tests/test_dataset.py tests/test_train.py -v`
Expected: the new tests FAIL (`TypeError: unexpected keyword argument 'normalization'`).

- [ ] **Step 3: Implement** `channel_stats`, the `EMGFK` argument and the `load_data` change. Add
  `normalization: subject` to `src/params.yaml`. In `scripts/build_dataset.py`, delete the z-score line and
  say in `main`'s docstring that normalization happens in `EMGFK`.

- [ ] **Step 4: Run and confirm everything passes**

Run: `pytest -v && ruff check . && ruff format --check .` → all PASS.

- [ ] **Step 5: Keep a copy of today's processed data**

Copy `data/processed/emg_fk/subject_01.npz` and `subject_19.npz` to the scratch directory, on the machine
that holds the data (locally or `hri-gpu`).

- [ ] **Step 6: Rebuild and check the real data**

Run: `dvc repro build_dataset && dvc push`. Then, for subjects 1 and 19, compare training examples 0, 100 and
the last one of `EMGFK(new_dir, [s], 1040, 40)` with the same slices of the saved copies:
`np.testing.assert_allclose(rtol=1e-5, atol=1e-5)`.
Expected: PASS. Also print each subject's per-channel std of the stored (un-normalized) EMG. If any std is
below 1e-4, report it to the user: `EPS = 1e-6` then shrinks the z-scored signal by `std / (std + 1e-6)`.
This is today's behaviour, and it is kept as it is.

- [ ] **Step 7: Update `README.md`** (normalization now happens at load time; the `normalization` key) **and commit**

```bash
git add scripts/build_dataset.py src/dataset.py scripts/train.py src/params.yaml README.md tests/ dvc.lock
git commit -m "feat: z-score at load time, per subject or pooled per set (normalization switch)"
```

---

### Task 13.2: Augmentation plumbing and A1 (`aug_gain`)

**Files:**
- Create: `src/augment.py`, `tests/test_augment.py`
- Modify: `src/dataset.py` (`EMGFK.__getitem__` applies `augment`), `scripts/train.py`, `src/params.yaml`,
  `dvc.yaml` (`src/augment.py` in `train` deps), `README.md`, `tests/test_train.py`

**Interfaces:**
- Produces in `src/augment.py`:
  - `channel_gain(x: Tensor[C, T], log_gain: Tensor[C]) -> Tensor`: `x * exp(log_gain)[:, None]`.
  - `class Augment(gain: float = 0.0, rotate: int = 0, shift: float = 0.0)`. Negative values raise
    `ValueError`. It owns a `torch.Generator` and never touches the global RNG.
    - `manual_seed(seed: int) -> None`.
    - `__call__(x: Tensor[8, T]) -> Tensor[8, T]`: draws its parameters per call, i.e. per training example.
      In this task: `log_gain = gain · N(0, 1)` per channel. Tasks 13.3–13.4 add rotation and shift, applied
      before the gain.
    - `from_params(params: dict) -> Augment | None` (staticmethod): `None` when `aug_gain`, `aug_rotate` and
      `aug_shift` are all 0.
- `EMGFK(..., augment=None)`: `__getitem__` returns `augment(x)` when `augment` is set.
- `scripts/train.py`: `load_data` gives `Augment.from_params(params)` to the training set only. `run` calls
  `train_set.augment.manual_seed(seed)` right after `torch.manual_seed(seed)` when augmentation is on.

- [ ] **Step 1: Write the failing tests**

```python
def test_gain_zero_is_identity():         # channel_gain(x, torch.zeros(8)) == x; from_params with all aug_* = 0 is None
def test_gain_is_per_channel_and_constant_over_time():
    aug = Augment(gain=0.2); aug.manual_seed(0)
    x = torch.rand(8, 240) + 0.5
    ratio = aug(x) / x
    assert torch.allclose(ratio, ratio[:, :1].expand_as(ratio))  # one gain per channel
    assert ratio[:, 0].unique().numel() == 8                      # different across channels


def test_gain_strength():                 # 2000 draws on ones(8, 1): std of log(ratio) within 0.2 ± 0.01
def test_augment_is_reproducible():       # two Augment(gain=0.2) seeded 3 give equal outputs on the same inputs
def test_augment_leaves_global_rng_alone():
    aug = Augment(gain=0.2); aug.manual_seed(0); x = torch.ones(8, 240)
    torch.manual_seed(1); a = torch.rand(3)
    torch.manual_seed(1); aug(x); b = torch.rand(3)
    assert torch.equal(a, b)


def test_negative_strengths_raise():      # Augment(gain=-0.1), Augment(rotate=-1), Augment(shift=-0.5) -> ValueError
```

In `tests/test_train.py`:

```python
def test_only_training_examples_are_augmented(data_dir):
    train_set, _, val_loader, test_loader = load_data({**PARAMS, "aug_gain": 0.2}, data_dir, 240)
    assert not torch.equal(train_set[0][0], train_set[0][0])
    for ds in (val_loader.dataset, test_loader.dataset):
        assert ds.augment is None and torch.equal(ds[0][0], ds[0][0])
```

plus `run` completing with `aug_gain: 0.2`.

- [ ] **Step 2: Run them and confirm they fail** (`ModuleNotFoundError: src.augment`).
- [ ] **Step 3: Implement**, and add `aug_gain: 0.0` (and, now, `aug_rotate: 0`, `aug_shift: 0.0`, which
  `from_params` reads) to `src/params.yaml`.
- [ ] **Step 4: Run and confirm everything passes:** `pytest -v && ruff check . && ruff format --check .`
- [ ] **Step 5: Update `README.md` (augmentation keys; training examples only) and commit**

```bash
git add src/augment.py src/dataset.py scripts/train.py src/params.yaml dvc.yaml README.md tests/
git commit -m "feat: seeded training-only augmentation, per-channel gain (aug_gain)"
```

---

### Task 13.3: A2 (`aug_rotate`): whole-electrode ring rotation

**Files:**
- Modify: `src/augment.py`, `tests/test_augment.py`, `README.md`

**Interfaces:**
- Produces in `src/augment.py`:
  - `RING_ORDER: tuple[int, ...]`: channel indices in order around the armband. Identity if Step 1
    confirms that `EMG 1`…`EMG 8` follow the ring.
  - `rotate(x: Tensor[8, T], r: int, order=RING_ORDER) -> Tensor`: moves the channel at ring position i to
    ring position i + r (mod 8). With the identity order this is `x.roll(r, 0)`.
  - `Augment(rotate=R)` draws r uniformly from {−R, …, R} (0 included) per call.

- [ ] **Step 1: Establish the ring order (blocks A2 and A3 only).** Print `raw.info["chs"][i]["loc"][:3]` for
  `EMG 1`…`EMG 8` of `Subject_01.fif`. If the positions are set, order the channels by angle around the
  forearm axis. If they are zero or NaN, check `docs/continuous_paper.pdf` §II and the dataset's description.
  If the order is still unknown, **stop and ask the user**: do not guess. Record the source in `RING_ORDER`'s
  comment.
- [ ] **Step 2: Write the failing tests**

```python
def test_rotate_is_roll_in_ring_order():
    x = torch.arange(8.0)[:, None].expand(8, 5)
    torch.testing.assert_close(rotate(x, 1, order=tuple(range(8))), x.roll(1, 0))
    order = (0, 2, 4, 6, 1, 3, 5, 7)
    y = rotate(x, 1, order=order)
    assert [int(y[c, 0]) for c in order] == [order[(i - 1) % 8] for i in range(8)]


def test_rotate_full_turn_is_identity():   # rotate(x, 8) == x; rotate(x, -1) == rotate(x, 7)
def test_rotation_draws_cover_minus_R_to_R():  # Augment(rotate=1), 300 draws on distinct channels: r in {-1, 0, 1}, each seen
```

- [ ] **Step 3: Run them and confirm they fail** (`ImportError: rotate`).
- [ ] **Step 4: Implement**, then run `pytest -v && ruff check . && ruff format --check .` → all PASS.
- [ ] **Step 5: Commit**

```bash
git add src/augment.py tests/test_augment.py README.md
git commit -m "feat: electrode-ring rotation augmentation (aug_rotate)"
```

---

### Task 13.4: A3 (`aug_shift`): fractional ring shift

**Files:**
- Modify: `src/augment.py`, `tests/test_augment.py`, `README.md`

**Interfaces:**
- Consumes: `rotate`, `RING_ORDER` (Task 13.3).
- Produces: `shift(x: Tensor[8, T], u: float, order=RING_ORDER) -> Tensor`. With `r = floor(u)` and
  `f = u − r`, it returns `(1 − f) · rotate(x, r) + f · rotate(x, r + 1)`: linear interpolation between ring
  neighbours. `Augment(shift=S)` draws u ~ U(−S, S) per call. The order within `__call__` is rotate, then
  shift, then gain.

Known effect, which A3 tests rather than corrects: interpolating two partly correlated channels lowers their
variance, by up to half at u = ±0.5 for uncorrelated neighbours.

- [ ] **Step 1: Write the failing tests**

```python
def test_shift_integer_values_are_rotations():  # shift(x, 0) == x; shift(x, 1) == rotate(x, 1); shift(x, -1) == rotate(x, -1)
def test_shift_interpolates_neighbours():
    x = torch.randn(8, 20)
    torch.testing.assert_close(shift(x, 0.25), 0.75 * x + 0.25 * rotate(x, 1))
    torch.testing.assert_close(shift(x, -0.5), 0.5 * x + 0.5 * rotate(x, -1))


def test_shift_draws_stay_in_range():           # Augment(shift=0.5) on a ramp input: recovered |u| <= 0.5 over 300 draws
```

- [ ] **Step 2: Run them and confirm they fail** (`ImportError: shift`).
- [ ] **Step 3: Implement**, then run `pytest -v && ruff check . && ruff format --check .` → all PASS, and
  `run` completes with `aug_shift: 0.5` (add it to the `test_train.py` run test).
- [ ] **Step 4: Commit**

```bash
git add src/augment.py tests/ README.md
git commit -m "feat: fractional electrode-ring shift augmentation (aug_shift)"
```

---

### Task 13.5: S1/S2 (`downsample`): keep every s-th tangent step after `ssm1`

**Files:**
- Modify: `src/model/ssm.py`, `src/model/model.py`, `scripts/train.py`, `src/params.yaml`, `README.md`
- Modify: `tests/test_model.py`, `tests/test_train.py`

**Interfaces:**
- Produces:
  - `SSMEncoder(..., downsample: int = 1)`. It raises `ValueError` if `downsample < 1`, if it does not divide
    `length`, or if `length // downsample < 2`. `ssm2` is built with length `length // downsample`. In
    `forward`, `ssm1`'s output is sliced `[:, downsample − 1 :: downsample]` before anything else in its
    block (MetaTF order: Kernelized → Stride → BatchNorm → activation → dropout). The last tangent step is
    always kept.
  - `Model(..., downsample: int = 1)` passes it to `SSMEncoder`. With Task 10's `readout="mean"`, the mean
    runs over the L / s remaining steps.
  - `scripts/train.py` builds `Model` with `downsample=params["downsample"]`; `src/params.yaml` gets
    `downsample: 1`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("L,s", [(10, 2), (10, 5), (50, 2), (50, 5)])
def test_downsample_keeps_output_shape(L, s):
    assert Model(8, 15, L, downsample=s)(torch.randn(4, 8, raw_length(L))).shape == (4, 15)


def test_downsample_feeds_every_sth_step_to_ssm2():
    enc = SSMEncoder(36, 10, "tanh", downsample=2).eval()
    seen = {}
    enc.ssm1.register_forward_hook(lambda m, i, o: seen.update(o1=o))
    enc.ssm2.register_forward_hook(lambda m, i, o: seen.update(i2=i[0]))
    enc(torch.randn(3, 10, 36))
    torch.testing.assert_close(seen["i2"], torch.tanh(seen["o1"][:, 1::2]))


def test_downsample_inits_ssm2_from_reduced_length():
    enc = SSMEncoder(36, 50, "tanh", downsample=5)
    torch.testing.assert_close(enc.ssm2.log_dt, Kernelized(256, 128, 10).log_dt)
    torch.testing.assert_close(enc.ssm1.log_dt, Kernelized(256, 256, 50).log_dt)


@pytest.mark.parametrize("L,s", [(10, 3), (10, 10), (10, 0)])
def test_bad_downsample_raises(L, s):
    with pytest.raises(ValueError):
        SSMEncoder(36, L, "tanh", downsample=s)


def test_downsample_1_is_baseline():        # same seed: Model(8,15,10) and Model(8,15,10, downsample=1) have equal state_dicts and outputs
```

In `tests/test_train.py`, `run` completes with `downsample: 2` and with `downsample: 5` (synthetic L = 10).

- [ ] **Step 2: Run them and confirm they fail** (`TypeError: unexpected keyword argument 'downsample'`).
- [ ] **Step 3: Implement**, then run `pytest -v && ruff check . && ruff format --check .` → all PASS.
- [ ] **Step 4: Update `README.md` and commit**

```bash
git add src/model scripts/train.py src/params.yaml README.md tests/
git commit -m "feat: tangent-step downsampling between Kernelized layers (downsample)"
```

---

### Task 13.6: S3/S4 (`ssm_speed`): smaller initial Δt

**Files:**
- Modify: `src/model/ssm.py`, `src/model/model.py`, `scripts/train.py`, `src/params.yaml`, `README.md`,
  `tests/test_model.py`, `tests/test_train.py`

**Interfaces:**
- Produces: `SSMEncoder(..., speed: float = 4)` passed as `speed=` to both `ssm1` and `ssm2`;
  `Model(..., ssm_speed: float = 4)` passes it as `speed`. `scripts/train.py` reads `params["ssm_speed"]`.
  Δt stays trainable; only its initial value changes (`Kernelized` init, spec §4.1).

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("speed", [1, 0.25])
def test_ssm_speed_shifts_log_dt(speed):
    m, base = Model(8, 15, 50, ssm_speed=speed), Model(8, 15, 50)
    for layer in ("ssm1", "ssm2"):
        new, old = getattr(m.encoder, layer), getattr(base.encoder, layer)
        torch.testing.assert_close(new.log_dt, old.log_dt + math.log(speed / 4))
        torch.testing.assert_close(new.A, old.A)
        torch.testing.assert_close(new.B, old.B)


def test_nonpositive_speed_raises():  # Model(8, 15, 10, ssm_speed=0) -> ValueError
```

Plus `run` completing with `ssm_speed: 1` and `0.25`.

- [ ] **Step 2: Run them and confirm they fail** (`TypeError: unexpected keyword argument 'ssm_speed'`).
  Once the argument exists, `Kernelized`'s `math.log(speed)` already raises `ValueError` for speed ≤ 0. No
  extra check is needed.
- [ ] **Step 3: Implement**, then run `pytest -v && ruff check . && ruff format --check .` → all PASS.
- [ ] **Step 4: Update `README.md` and commit**

```bash
git add src/model scripts/train.py src/params.yaml README.md tests/
git commit -m "feat: Kernelized speed switch for a smaller initial dt (ssm_speed)"
```

---

### Task 13.7: S5/S6 (`ssm_lr_scale`, `ssm_decay_A`): parameter groups for the SSM dynamics

**Files:**
- Modify: `src/trainer.py`, `scripts/train.py`, `src/params.yaml`, `README.md`, `tests/test_trainer.py`,
  `tests/test_train.py`

**Interfaces:**
- Produces in `src/trainer.py`:
  `param_groups(model, lr: float, weight_decay: float, ssm_lr_scale: float = 1.0, ssm_decay_A: bool = True) -> list[dict]`.
  The SSM dynamics are `A` and `log_dt` of every `Kernelized` module (found by type, not by name). Each
  parameter gets `lr × ssm_lr_scale` if it is a dynamics parameter (else `lr`), and `weight_decay = 0` if it is
  an `A` and `ssm_decay_A` is false (else `weight_decay`). Parameters with equal settings share a group, and
  empty groups are left out, so the defaults give exactly one group. `ssm_lr_scale <= 0` raises `ValueError`.
- `scripts/train.py`: `torch.optim.AdamW(param_groups(model, params["lr"], params["weight_decay"],
  params["ssm_lr_scale"], params["ssm_decay_A"]))`. Schedulers scale each group from its own initial LR, so
  the ratio holds.

- [ ] **Step 1: Write the failing tests**

```python
def dynamics(m):
    return {id(p) for k in m.modules() if isinstance(k, Kernelized) for p in (k.A, k.log_dt)}


def test_param_groups_default_is_one_group():
    m = Model(8, 15, 10)
    (g,) = param_groups(m, 1e-3, 1e-4)
    assert (g["lr"], g["weight_decay"]) == (1e-3, 1e-4)
    assert {id(p) for p in g["params"]} == {id(p) for p in m.parameters()}


@pytest.mark.parametrize("scale,decay_A", [(0.1, True), (1.0, False), (0.1, False)])
def test_param_groups_cover_every_parameter_once(scale, decay_A):
    m = Model(8, 15, 10)
    ids = [id(p) for g in param_groups(m, 1e-3, 1e-4, scale, decay_A) for p in g["params"]]
    assert sorted(ids) == sorted(id(p) for p in m.parameters())


def test_ssm_lr_scale():
    m = Model(8, 15, 10)
    by_id = {id(p): g for g in param_groups(m, 1e-3, 1e-4, ssm_lr_scale=0.1) for p in g["params"]}
    for i, g in by_id.items():
        assert g["lr"] == pytest.approx(1e-4 if i in dynamics(m) else 1e-3) and g["weight_decay"] == 1e-4


def test_no_decay_on_A():               # ssm_decay_A=False: every Kernelized A has weight_decay 0; log_dt and the rest 1e-4; all lr 1e-3
def test_ssm_lr_ratio_survives_cosine():
    m = Model(8, 15, 10)
    opt = torch.optim.AdamW(param_groups(m, 1e-3, 1e-4, ssm_lr_scale=0.1))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 15)
    for _ in range(5):
        opt.step(); sched.step()
    lrs = sorted(g["lr"] for g in opt.param_groups)
    assert lrs[0] / lrs[1] == pytest.approx(0.1)
```

Plus `run` completing with `ssm_lr_scale: 0.1` and with `ssm_decay_A: false`.

- [ ] **Step 2: Run them and confirm they fail** (`ImportError: param_groups`).
- [ ] **Step 3: Implement**, add both keys to `src/params.yaml`, then run
  `pytest -v && ruff check . && ruff format --check .` → all PASS.
- [ ] **Step 4: Update `README.md` and commit**

```bash
git add src/trainer.py scripts/train.py src/params.yaml README.md tests/
git commit -m "feat: SSM dynamics LR scale and no-decay-on-A parameter groups"
```

---

### Task 13.8: Extra sweep and summary

**Files:**
- Create: `scripts/sweep_extra.sh`, `tests/test_sweep_extra.py`
- Modify: `scripts/summarize_sweep.py`, `tests/test_summarize_sweep.py`, `sky/train.yaml`, `README.md`

**Interfaces:**
- `scripts/sweep_extra.sh ACTIVATION K STRIDE`:
  - The same `dvc pull --allow-missing` / `dvc repro build_dataset` / `dvc push` preamble as `sweep.sh`.
  - 11 `dvc exp run -n x-$1-k$2-s$3-<label>` runs. Each passes `-S src/params.yaml:train.activation=$1`,
    `train.k=$2`, `train.example_stride=$3`, one `-S` per `BASE_SETS` entry, and its change:

    | label | change |
    |---|---|
    | `baseline` | none |
    | `aug_gain` | `aug_gain=0.2` |
    | `aug_rotate` | `aug_rotate=1` |
    | `aug_shift` | `aug_shift=0.5` |
    | `norm_set` | `normalization=set` |
    | `downsample_2`, `downsample_5` | `downsample=2`, `downsample=5` |
    | `speed_1`, `speed_0p25` | `ssm_speed=1`, `ssm_speed=0.25` |
    | `ssm_lr_0p1` | `ssm_lr_scale=0.1` |
    | `no_decay_a` | `ssm_decay_A=false` |

  - Env `BASE_SETS`: space-separated `key=value` overrides of `train` keys (e.g. Section 4 winners such as
    `"weight_decay=0.05 lr_schedule=warmup_cosine"`). If any key is `activation`, `k`, `example_stride` or a
    key from the table, the script exits 1 before running anything.
  - Env `DRY_RUN=1`: prints the 11 `dvc exp run …` commands, one per line, and runs nothing (no preamble).
  - Writes `dvc exp show --csv > extra_$1_k$2_s$3.csv` and honours `PUSH_EXPERIMENTS` like `sweep.sh`. An
    existing experiment name makes `dvc exp run` fail. The script does not overwrite: remove old runs with
    `dvc exp remove` first.
- `scripts/summarize_sweep.py`:
  - `read_rows(path, columns=COLUMNS)` (the same signature as Task 11's).
    `EXTRA_COLUMNS = ("Experiment", VAL, "val.mae.std", TEST)`.
  - `base_from_filename(path) -> tuple[str, str, str] | None` for `extra_<activation>_k<K>_s<STRIDE>.csv`.
  - `extra_rows(rows, activation, k, stride) -> dict[str, dict]`: label → row for experiments named
    `x-<activation>-k<k>-s<stride>-<label>` (the first listed wins). Raises `ValueError` if there is no
    `baseline`.
  - `beats_baseline(row, base) -> bool`: `base[VAL] − row[VAL] > base["val.mae.std"]`.
  - CLI on an `extra_*.csv`: a table sorted by val MAE (label, val mean ± std, Δ val, test, Δ test, `*` where
    `beats_baseline`), then the label with the lowest val MAE and its test verdict. Test never ranks.
- `sky/train.yaml`: env `EXTRA: ""` (e.g. `"tanh 5 40"`) and `BASE_SETS: ""`. If `EXTRA` is set, it runs
  `bash scripts/sweep_extra.sh $EXTRA`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sweep_extra.py (skipped when shutil.which("bash") is None)
def test_sweep_extra_dry_run():
    out = dry_run("tanh", "5", "40")  # DRY_RUN=1 bash scripts/sweep_extra.sh ..., returns stdout lines
    assert len(out) == 11 and all(line.startswith("dvc exp run -n x-tanh-k5-s40-") for line in out)
    assert all("train.activation=tanh" in l and "train.k=5" in l and "train.example_stride=40" in l for l in out)
    assert sum("train.aug_gain=0.2" in l for l in out) == 1


def test_sweep_extra_passes_base_sets():   # BASE_SETS="weight_decay=0.05": every line has -S src/params.yaml:train.weight_decay=0.05
def test_sweep_extra_rejects_conflicting_base():  # BASE_SETS="downsample=2" and "k=1": exit code 1, nothing printed on stdout


# tests/test_summarize_sweep.py
def test_extra_rows_by_experiment_name(tmp_path):
    # rows: workspace; x-tanh-k5-s40-baseline; x-tanh-k5-s40-aug_gain; x-tanh-k1-s40-aug_gain (other base)
    assert set(extra_rows(read_rows(path, EXTRA_COLUMNS), "tanh", "5", "40")) == {"baseline", "aug_gain"}


def test_extra_rows_without_baseline_raise(tmp_path):
def test_beats_baseline():   # base 17.0 ± 0.3: 16.6 beats, 16.8 does not
def test_extra_selection_ignores_test(tmp_path):  # better test but worse val is not the selected label
```

- [ ] **Step 2: Run them and confirm they fail**, then **implement**, then
  `pytest -v && ruff check . && ruff format --check .` → all PASS.
- [ ] **Step 3: Local end-to-end check of one switch.** Run
  `dvc exp run -n x-check -S src/params.yaml:train.ssm_decay_A=false -S src/params.yaml:train.epochs=1 -S "src/params.yaml:train.seeds=[0]"`
  on the local data. Confirm that `dvc exp show` lists `ssm_decay_A` as `false` and the run finishes. Then
  run `dvc exp remove x-check`.
- [ ] **Step 4: Update `README.md` (how to run and summarize the extra sweep) and commit**

```bash
git add scripts/sweep_extra.sh scripts/summarize_sweep.py tests/ sky/train.yaml README.md
git commit -m "feat: augmentation, normalization and SSM regularization sweep with named experiments"
```

---

### Task 13.9: Run the extra sweep (needs the user's go-ahead: GPU cost)

- [ ] **Step 1: Get the base configuration** (activation, k, example stride and any `BASE_SETS`) from the
  user: Task 12 Step 5 if Section 4 ran, otherwise Task 9 Step 5.
- [ ] **Step 2: Confirm that Task 13.3 Step 1 established the ring order.** If it did not, the user decides
  whether to run without A2/A3.
- [ ] **Step 3: Ask the user to confirm 33 GPU runs** (11 experiments × 3 seeds) per base configuration on
  SkyPilot. Do not launch without a yes.
- [ ] **Step 4: Run.** `sky exec hri-gpu sky/train.yaml --env EXTRA="<activation> <k> <stride>" --env BASE_SETS="<…>"`,
  then `rsync hri-gpu:sky_workdir/extra_<activation>_k<k>_s<stride>.csv .`, then
  `python scripts/summarize_sweep.py extra_<activation>_k<k>_s<stride>.csv`.
- [ ] **Step 5: Report to the user:** the table, which changes beat the baseline's val MAE by more than its
  seed std, and the val-selected change's test MAE and verdict against 16.71° / 17.21°. Combining winners is
  the user's call and is not part of this plan.
