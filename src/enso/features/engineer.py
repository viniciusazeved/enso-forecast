"""Engenharia de features alinhada por instante t com lags >= 1.

Toda feature em t depende SOMENTE de informacao disponivel ate t (passado).
Sem janelas centradas. Sem rolling com center=True. Encoding sazonal eh permitido
(funcao deterministica do mes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_lagged_features(
    df: pd.DataFrame,
    cols: list[str],
    lags: list[int],
) -> pd.DataFrame:
    """Cria colunas '<col>_lag<k>' para cada k em lags."""
    out = {}
    for c in cols:
        s = df[c]
        for k in lags:
            if k < 1:
                raise ValueError(f"lag deve ser >= 1, recebido {k}")
            out[f"{c}_lag{k}"] = s.shift(k)
    return pd.DataFrame(out, index=df.index)


def make_rolling_features(
    df: pd.DataFrame,
    cols: list[str],
    windows: list[int],
    stats: tuple[str, ...] = ("mean", "std"),
    base_lag: int = 1,
) -> pd.DataFrame:
    """Rolling stats SEM centro, com base_lag >= 1 (informacao ate t - base_lag)."""
    out = {}
    for c in cols:
        s_base = df[c].shift(base_lag)
        for w in windows:
            if "mean" in stats:
                out[f"{c}_rmean{w}_lag{base_lag}"] = s_base.rolling(w, min_periods=w).mean()
            if "std" in stats:
                out[f"{c}_rstd{w}_lag{base_lag}"]  = s_base.rolling(w, min_periods=w).std()
    return pd.DataFrame(out, index=df.index)


def make_seasonal_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Encoding sin/cos do mes (funcao deterministica do tempo, sem leakage)."""
    months = index.month.values.astype(float)
    return pd.DataFrame(
        {
            "sin_month": np.sin(2 * np.pi * months / 12.0),
            "cos_month": np.cos(2 * np.pi * months / 12.0),
        },
        index=index,
    )


def build_feature_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    lags: list[int] = (1, 2, 3, 6, 12),
    rolling_windows: list[int] = (3, 6, 12),
    seasonal: bool = True,
) -> pd.DataFrame:
    """Monta a matriz X completa (sem dropna) - o caller decide como tratar NaNs."""
    pieces: list[pd.DataFrame] = []
    pieces.append(make_lagged_features(df, list(feature_cols), list(lags)))
    if rolling_windows:
        pieces.append(
            make_rolling_features(df, list(feature_cols), list(rolling_windows), base_lag=1)
        )
    if seasonal:
        pieces.append(make_seasonal_features(df.index))
    X = pd.concat(pieces, axis=1)
    return X


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    horizon: int,
    lookback: int = 12,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Janelamento sequencial para modelos com input [batch, seq, features].

    Para origem t, retorna:
      X[i] = df[feature_cols].iloc[t-lookback+1 .. t].values  (forma: lookback, F)
      y[i] = df[target_col].iloc[t + horizon]                  (escalar)

    Indice retornado eh o array de timestamps de origem t (instante de previsao).
    Drops linhas com NaN em features ou alvo.
    """
    if lookback < 1:
        raise ValueError("lookback deve ser >= 1")
    if horizon < 1:
        raise ValueError("horizon deve ser >= 1")
    feats = df[feature_cols].values
    targs = df[target_col].values
    dates = df.index
    Xs, ys, ds = [], [], []
    n = len(df)
    for t in range(lookback - 1, n - horizon):
        win = feats[t - lookback + 1 : t + 1]      # [lookback, F]
        y   = targs[t + horizon]
        if np.isnan(win).any() or np.isnan(y):
            continue
        Xs.append(win)
        ys.append(y)
        ds.append(dates[t])
    if not Xs:
        raise ValueError("Nenhuma janela valida; verifique NaNs nas features.")
    return np.stack(Xs), np.array(ys), pd.DatetimeIndex(ds)


def make_supervised(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    horizon: int,
    lags: list[int] = (1, 2, 3, 6, 12),
    rolling_windows: list[int] = (3, 6, 12),
    seasonal: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Monta (X, y) para horizonte de previsao h passos a frente.

    Convencao: y(t) = target(t + h). Logo, em t precisamos de features
    construidas com info ate t. Aqui shiftamos o alvo *para cima* por h, e o
    indice de y eh o instante de origem da previsao (t).
    """
    if horizon < 1:
        raise ValueError("horizon deve ser >= 1")
    X = build_feature_matrix(
        df,
        feature_cols=list(feature_cols),
        lags=list(lags),
        rolling_windows=list(rolling_windows),
        seasonal=seasonal,
    )
    y = df[target_col].shift(-horizon).rename(f"{target_col}_h{horizon}")
    aligned = X.join(y, how="inner").dropna()
    return aligned.drop(columns=[y.name]), aligned[y.name]
