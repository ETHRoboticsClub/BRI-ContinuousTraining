# Raw Datasets (DVC Tracked)

## 1. EMG-FK 200 Hz Processed Recordings (`data/raw/emg_fk.dvc`)

- **Dataset**: EMG Finger-Kinematics (Colot et al., arXiv:2604.22499, 2026)
- **Source**: Zenodo record [19453843](https://zenodo.org/records/19453843)
- **Sampling Frequency**: Downsampled to **200 Hz** (original: 500 Hz) using polyphase rational resampling (`scipy.signal.resample_poly(up=2, down=5)`) with an exact Kaiser FIR anti-aliasing lowpass filter (cutoff at the 100 Hz Nyquist frequency).
- **Subjects Included**: **5 subjects** (Subjects 01 through 05).
- **Recordings Count**: 10 `.npz` files (2 trials per subject, ~15.1 minutes per recording, 181,290 samples each).
- **Input Channels**: 8 forearm surface EMG electrodes.
- **Regression Targets**: 15 continuous finger joint angles (degrees).
- **Metadata**: `meta_subject`, `meta_session`, `meta_trial`.

## 2. EMG-FK Source Data (`data/raw/emg_fk_source.dvc`)

- **Format**: Raw MNE `.fif` recordings directly downloaded from Zenodo.
- **Sampling Rate**: Original 500 Hz.
- **Subjects**: 5 files (`Subject_01.fif` to `Subject_05.fif`, ~105 MB each).
