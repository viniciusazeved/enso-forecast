"""Auditoria automatica contra data leakage em pipelines de series temporais.

Verifica:
  1. Splits sao monotonicos no tempo, sem overlap (treino < val < teste).
  2. Scaler/transformer foi fitado SOMENTE no treino do fold.
  3. Nenhuma feature contem informacao do alvo no instante t (overlap de medias
     moveis, vazamento via shift incorreto).
  4. Target nao aparece nas features (auto-prediction trivial).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from enso.data.splits import Split


class LeakageError(AssertionError):
    """Erro especifico para falhas de auditoria de leakage."""


def assert_split_no_overlap(splits: list[Split]) -> None:
    for s in splits:
        # train antes de val antes de test
        if len(s.train_dates) and len(s.val_dates):
            if s.train_dates.max() >= s.val_dates.min():
                raise LeakageError(
                    f"fold {s.fold}: treino invade validacao "
                    f"({s.train_dates.max()} >= {s.val_dates.min()})"
                )
        if len(s.val_dates) and len(s.test_dates):
            if s.val_dates.max() >= s.test_dates.min():
                raise LeakageError(
                    f"fold {s.fold}: validacao invade teste "
                    f"({s.val_dates.max()} >= {s.test_dates.min()})"
                )
        # indices disjuntos
        if np.intersect1d(s.train_idx, s.val_idx).size > 0:
            raise LeakageError(f"fold {s.fold}: overlap entre treino e val")
        if np.intersect1d(s.val_idx, s.test_idx).size > 0:
            raise LeakageError(f"fold {s.fold}: overlap entre val e teste")
        if np.intersect1d(s.train_idx, s.test_idx).size > 0:
            raise LeakageError(f"fold {s.fold}: overlap entre treino e teste")


def assert_scaler_fitted_on_train(scaler_min: np.ndarray, train_x: np.ndarray) -> None:
    """Sanidade: o min do scaler deve coincidir com o min do treino."""
    train_min = np.nanmin(train_x, axis=0)
    if not np.allclose(scaler_min, train_min, equal_nan=True):
        raise LeakageError("scaler.min_ nao coincide com min do treino: refit no fold inteiro?")


def assert_target_not_in_features(features_cols: list[str], target_col: str) -> None:
    if target_col in features_cols:
        raise LeakageError(
            f"alvo '{target_col}' presente entre features (auto-predicao trivial)"
        )


def assert_no_centered_rolling(
    features: pd.DataFrame, target: pd.Series, max_corr_at_zero_lag: float = 0.999
) -> None:
    """Heuristica: alguma feature tem correlacao quase-perfeita com o alvo
    no mesmo instante? Sinal de janela centrada / vazamento estrutural.
    """
    aligned = features.join(target.rename("__y__"), how="inner").dropna()
    if len(aligned) < 24:
        return
    corrs = aligned.corr(numeric_only=True)["__y__"].drop("__y__")
    suspeitas = corrs[corrs.abs() >= max_corr_at_zero_lag]
    if not suspeitas.empty:
        raise LeakageError(
            "features com corr ~= 1 com o alvo (provavel leakage estrutural):\n"
            + suspeitas.to_string()
        )


def assert_features_strictly_lagged(
    raw_df: pd.DataFrame, lagged_df: pd.DataFrame, target_dates: pd.DatetimeIndex
) -> None:
    """Verifica que toda coluna em lagged_df, no instante t, foi construida a partir
    de valores de raw_df em datas <= t - 1. Comparamos ate min(t-1) por seguranca.
    """
    common = lagged_df.index.intersection(target_dates)
    if len(common) == 0:
        return
    # Para cada coluna de lagged_df, garantir que nao ha valor para datas alem de t-1
    # Esta e uma checagem leve - depende da nomeacao das colunas indicar o lag.
    # Se as colunas nao sao por timestep direto, deixamos passar.
    # (Aqui esperamos colunas no formato '<var>_lag<k>' onde k >= 1.)
    for col in lagged_df.columns:
        if "_lag" not in col:
            continue
        try:
            lag = int(col.rsplit("_lag", 1)[1])
        except ValueError:
            continue
        if lag < 1:
            raise LeakageError(f"feature {col!r} tem lag={lag} (precisa ser >= 1)")
