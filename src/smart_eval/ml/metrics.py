"""Evaluation metric helpers for regression runs."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, R2, RMSE, and MAPE (safe for zero targets)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)
    if np.isnan(mape):
        mape = 0.0

    return {
        "mae": mae,
        "r2": r2,
        "rmse": rmse,
        "mape": mape,
    }
