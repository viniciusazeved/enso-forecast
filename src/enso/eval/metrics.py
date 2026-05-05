"""Metricas de avaliacao para forecasting climatico."""
from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Anomaly Correlation Coefficient. Em climatologia, == Pearson sobre anomalias."""
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    denom = float(np.sqrt(np.sum(yt ** 2) * np.sum(yp ** 2)))
    return float(np.sum(yt * yp) / denom) if denom > 0 else float("nan")


def pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float = 0.5) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def hit_rate_phase(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Concordancia de fase ENSO (El Nino/Neutro/La Nina) com limiar +/- 0.5."""
    def phase(x):
        ph = np.zeros_like(x, dtype=int)
        ph[x >=  threshold] =  1
        ph[x <= -threshold] = -1
        return ph
    return float(np.mean(phase(y_true) == phase(y_pred)))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae":   mae(y_true, y_pred),
        "rmse":  rmse(y_true, y_pred),
        "r2":    r2(y_true, y_pred),
        "acc":   acc(y_true, y_pred),
        "hit":   hit_rate_phase(y_true, y_pred),
    }
