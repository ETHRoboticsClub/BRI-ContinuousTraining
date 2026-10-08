import numpy as np


def regression_metrics(y_true, y_pred):
    """y_true, y_pred: [N, D]. Rows must be in time order for jitter_ratio."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    var = y_true.var(axis=0)
    r2 = 1 - (err**2).mean(axis=0) / np.where(var > 0, var, np.nan)
    t, p = y_true - y_true.mean(axis=0), y_pred - y_pred.mean(axis=0)
    pearson = (t * p).sum(axis=0) / np.sqrt((t**2).sum(axis=0) * (p**2).sum(axis=0) + 1e-12)
    # > 1: the prediction moves more than the target, i.e. the arm shakes
    jitter = np.abs(np.diff(y_pred, axis=0)).mean() / (np.abs(np.diff(y_true, axis=0)).mean() + 1e-12)
    return {
        "mse": float((err**2).mean()),
        "mae": float(np.abs(err).mean()),
        "r2": float(np.nanmean(r2)),
        "pearson": float(pearson.mean()),
        "jitter_ratio": float(jitter),
        "r2_per_target": r2.tolist(),
    }
