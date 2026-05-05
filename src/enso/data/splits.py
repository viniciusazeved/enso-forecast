"""Walk-forward (expanding window) CV para series temporais.

Garante que treino sempre antecede validacao/teste e jamais ha embaralhamento.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Split:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex


def walk_forward(
    dates: pd.DatetimeIndex,
    n_folds: int = 5,
    val_size: int = 24,
    test_size: int = 24,
    min_train_size: int = 120,
) -> Iterator[Split]:
    """Gera n_folds com janelas expansivas (cada fold cresce o treino).

    Layout do fold k (k = 0..n_folds-1):
        train: [0, train_end_k)
        val:   [train_end_k, train_end_k + val_size)
        test:  [train_end_k + val_size, train_end_k + val_size + test_size)

    train_end_k cresce em passos uniformes para que o ultimo fold termine no
    fim da serie. Garante: min_train_size <= treino e nenhum overlap.
    """
    n = len(dates)
    last_train_end = n - val_size - test_size
    if last_train_end < min_train_size:
        raise ValueError(
            f"Serie muito curta ({n}) para n_folds={n_folds}, val={val_size}, test={test_size}"
        )

    if n_folds == 1:
        train_ends = [last_train_end]
    else:
        train_ends = np.linspace(min_train_size, last_train_end, n_folds, dtype=int).tolist()

    for k, te in enumerate(train_ends):
        tr = np.arange(0, te)
        va = np.arange(te, te + val_size)
        ts = np.arange(te + val_size, te + val_size + test_size)
        yield Split(
            fold=k,
            train_idx=tr,
            val_idx=va,
            test_idx=ts,
            train_dates=dates[tr],
            val_dates=dates[va],
            test_dates=dates[ts],
        )


def summarize_splits(splits: list[Split]) -> pd.DataFrame:
    rows = []
    for s in splits:
        rows.append({
            "fold": s.fold,
            "train_inicio": s.train_dates.min(),
            "train_fim":    s.train_dates.max(),
            "n_train":      len(s.train_idx),
            "val_inicio":   s.val_dates.min(),
            "val_fim":      s.val_dates.max(),
            "n_val":        len(s.val_idx),
            "test_inicio":  s.test_dates.min(),
            "test_fim":     s.test_dates.max(),
            "n_test":       len(s.test_idx),
        })
    return pd.DataFrame(rows)
