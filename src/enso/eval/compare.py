"""Agregacao e ranking dos resultados por modelo x horizonte."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


METRIC_DIRECTION = {
    "mae": "min",
    "rmse": "min",
    "r2": "max",
    "acc": "max",
    "hit": "max",
}


def load_metrics(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(run_dir) / "metrics.csv")


def load_predictions(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(run_dir) / "predictions.csv", parse_dates=["date"])
    return df


def aggregate(df_m: pd.DataFrame) -> pd.DataFrame:
    """Media e desvio por (model, horizon) sobre folds e seeds."""
    agg = (
        df_m.groupby(["model", "horizon"])[["mae", "rmse", "r2", "acc", "hit"]]
        .agg(["mean", "std"])
    )
    return agg


def unified_score(df_m: pd.DataFrame) -> pd.DataFrame:
    """Score unificado normalizado (0..1) usando metricas disponiveis.

    Considera mae, rmse (menor eh melhor) e r2, acc (maior eh melhor).
    Metricas com todos os valores NaN sao ignoradas. Pra cada horizonte,
    normaliza entre os modelos e tira media simples.
    """
    cols = [c for c in ["mae", "rmse", "r2", "acc"] if c in df_m.columns]
    df = df_m.groupby(["model", "horizon"])[cols].mean().reset_index()
    out = []
    for h, sub in df.groupby("horizon"):
        s = sub.copy()
        valid_norm_cols = []
        for col in ["mae", "rmse"]:
            if col not in s.columns or s[col].isna().all():
                continue
            x = s[col]
            rng = x.max() - x.min()
            s[f"{col}_n"] = 1.0 - (x - x.min()) / rng if rng > 0 else 1.0
            valid_norm_cols.append(f"{col}_n")
        for col in ["r2", "acc"]:
            if col not in s.columns or s[col].isna().all():
                continue
            x = s[col]
            rng = x.max() - x.min()
            s[f"{col}_n"] = (x - x.min()) / rng if rng > 0 else 1.0
            valid_norm_cols.append(f"{col}_n")
        s["score"] = s[valid_norm_cols].mean(axis=1) if valid_norm_cols else 0.0
        out.append(s)
    return pd.concat(out, ignore_index=True)


def winner_per_horizon(df_m: pd.DataFrame) -> pd.DataFrame:
    """Eleva o vencedor por horizonte segundo score unificado."""
    s = unified_score(df_m)
    return s.sort_values(["horizon", "score"], ascending=[True, False]).groupby("horizon").head(1)


def diebold_mariano(
    e1: np.ndarray, e2: np.ndarray, h: int = 1
) -> tuple[float, float]:
    """Teste Diebold-Mariano (loss = abs error) entre dois modelos.

    Retorna (estatistica, p-valor aprox via t-Student n-1 graus).
    Hipotese nula: igual acuracia preditiva.
    """
    from scipy import stats

    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    if n < 5:
        return float("nan"), float("nan")
    mean = float(np.mean(d))
    # variancia HAC simples (Newey-West com lag h-1)
    gamma0 = float(np.var(d, ddof=0))
    var = gamma0
    for k in range(1, h):
        gk = float(np.cov(d[:-k], d[k:], bias=True)[0, 1])
        var += 2 * (1 - k / h) * gk
    if var <= 0:
        return float("nan"), float("nan")
    dm_stat = mean / np.sqrt(var / n)
    p = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    return float(dm_stat), float(p)


def pairwise_dm(
    df_p: pd.DataFrame, horizon: int, baseline: str = "persistence"
) -> pd.DataFrame:
    """Para um horizonte, roda DM de cada modelo vs baseline em cada fold/seed."""
    sub = df_p[df_p["horizon"] == horizon]
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for model, g in sub.groupby("model"):
        if model == baseline:
            continue
        # alinhar por (fold, seed, date) com o baseline
        b = sub[sub["model"] == baseline][["fold", "seed", "date", "y_true", "y_pred"]] \
            .rename(columns={"y_pred": "y_base"})
        merged = g.merge(b, on=["fold", "seed", "date", "y_true"])
        if merged.empty:
            continue
        e_model = merged["y_true"].values - merged["y_pred"].values
        e_base  = merged["y_true"].values - merged["y_base"].values
        stat, p = diebold_mariano(e_model, e_base, h=horizon)
        rows.append({"model": model, "horizon": horizon, "vs": baseline,
                     "dm": stat, "p": p, "n": len(merged)})
    return pd.DataFrame(rows)
