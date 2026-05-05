"""Trainer multi-modelo, multi-seed, multi-horizonte com walk-forward CV.

Output: DataFrame longo com colunas
  [model, horizon, fold, seed, split, mae, rmse, r2, acc, hit]

E DataFrame de previsoes longo:
  [model, horizon, fold, seed, date, y_true, y_pred]
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from sklearn.preprocessing import StandardScaler

from enso.config import HORIZONS, TARGET_COL
from enso.data.leakage import assert_split_no_overlap
from enso.data.loader import CORE_FEATURES, load_master, select_features
from enso.data.splits import Split, walk_forward
from enso.eval.metrics import all_metrics
from enso.features.engineer import make_sequences
from enso.models.base import ForecastModel
from enso.models.baselines import Climatology, DLinear, Persistence, SARIMA, SeasonalNaive
from enso.models.dl import LSTM, MLP, TCN, Mamba, Transformer

console = Console()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    horizons: tuple[int, ...] = tuple(HORIZONS)
    seeds: tuple[int, ...] = tuple(range(10))
    n_folds: int = 5
    val_size: int = 24
    test_size: int = 24
    min_train_size: int = 180
    lookback: int = 12
    since: str = "1982-01-01"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 200
    batch_size: int = 64
    lr: float = 1e-3
    include_models: tuple[str, ...] = (
        "persistence", "seasonal_naive", "climatology", "sarima", "dlinear",
        "mlp", "lstm", "tcn", "transformer", "mamba",
    )
    verbose: bool = False


def _build_models(cfg: TrainConfig, n_features: int, horizon: int, target_idx: int) -> list[ForecastModel]:
    L = cfg.lookback
    models: list[ForecastModel] = []
    if "persistence" in cfg.include_models:
        models.append(Persistence(target_col_idx=target_idx))
    if "seasonal_naive" in cfg.include_models:
        models.append(SeasonalNaive(target_col_idx=target_idx, horizon=horizon))
    if "climatology" in cfg.include_models:
        models.append(Climatology(target_col_idx=target_idx))
    if "sarima" in cfg.include_models:
        models.append(SARIMA(target_col_idx=target_idx, horizon=horizon))
    if "dlinear" in cfg.include_models:
        models.append(DLinear(lookback=L, horizon=horizon, target_col_idx=target_idx,
                              epochs=cfg.epochs, lr=cfg.lr, device=cfg.device))
    if "mlp" in cfg.include_models:
        models.append(MLP(lookback=L, n_features=n_features,
                          epochs=cfg.epochs, lr=cfg.lr, batch_size=cfg.batch_size, device=cfg.device))
    if "lstm" in cfg.include_models:
        models.append(LSTM(n_features=n_features, hidden=64, layers=2, dropout=0.2,
                           epochs=cfg.epochs, lr=cfg.lr, batch_size=cfg.batch_size, device=cfg.device))
    if "tcn" in cfg.include_models:
        models.append(TCN(n_features=n_features, channels=(64, 64, 64), k=3, dropout=0.1,
                          epochs=cfg.epochs, lr=cfg.lr, batch_size=cfg.batch_size, device=cfg.device))
    if "transformer" in cfg.include_models:
        models.append(Transformer(n_features=n_features, d_model=64, nhead=4, num_layers=2,
                                  dim_ff=128, dropout=0.1,
                                  epochs=cfg.epochs, lr=cfg.lr, batch_size=cfg.batch_size, device=cfg.device))
    if "mamba" in cfg.include_models:
        models.append(Mamba(n_features=n_features, d_model=64, n_layers=2, d_state=16,
                            d_conv=4, expand=2,
                            epochs=cfg.epochs, lr=cfg.lr, batch_size=cfg.batch_size, device=cfg.device))
    return models


def _scale_per_fold(X_train, X_val, X_test):
    """StandardScaler ajustado SO no treino, aplicado em [N, L, F] reshapeando."""
    N, L, F = X_train.shape
    sc = StandardScaler().fit(X_train.reshape(-1, F))
    def _t(x): return sc.transform(x.reshape(-1, F)).reshape(x.shape)
    return _t(X_train), _t(X_val), _t(X_test), sc


def run(cfg: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    console.print(f"[bold cyan]Device:[/bold cyan] {cfg.device}")
    df = load_master(since=cfg.since)
    df = select_features(df, target_col=TARGET_COL)

    feature_cols = [c for c in CORE_FEATURES if c in df.columns and c != TARGET_COL]
    if TARGET_COL not in feature_cols:
        feature_cols = [TARGET_COL] + feature_cols  # target_idx=0 garantido

    target_idx = feature_cols.index(TARGET_COL)
    n_features = len(feature_cols)
    console.print(f"features={n_features} (target_idx={target_idx})")

    metric_rows: list[dict] = []
    pred_rows: list[dict] = []

    for h in cfg.horizons:
        console.rule(f"[bold yellow]horizonte {h}m")
        X_seq, y_seq, dates = make_sequences(
            df, feature_cols=feature_cols, target_col=TARGET_COL,
            horizon=h, lookback=cfg.lookback,
        )
        console.print(f"X_seq={X_seq.shape}, y_seq={y_seq.shape}, periodo: {dates.min().date()} -> {dates.max().date()}")

        splits = list(walk_forward(
            dates=dates, n_folds=cfg.n_folds,
            val_size=cfg.val_size, test_size=cfg.test_size,
            min_train_size=cfg.min_train_size,
        ))
        assert_split_no_overlap(splits)

        for fold in splits:
            Xtr, Xva, Xts = X_seq[fold.train_idx], X_seq[fold.val_idx], X_seq[fold.test_idx]
            ytr, yva, yts = y_seq[fold.train_idx], y_seq[fold.val_idx], y_seq[fold.test_idx]
            dts = dates[fold.test_idx]

            Xtr_s, Xva_s, Xts_s, sc = _scale_per_fold(Xtr, Xva, Xts)

            # alvo do mes do teste para Climatology
            months_train = pd.DatetimeIndex(dates[fold.train_idx]).month.values
            months_test  = pd.DatetimeIndex(dts).month.values
            # ajusta para mes-do-alvo (t + h)
            months_train_target = ((months_train + h - 1) % 12) + 1
            months_test_target  = ((months_test  + h - 1) % 12) + 1

            for seed in cfg.seeds:
                set_seed(seed)
                models = _build_models(cfg, n_features, h, target_idx)
                for model in models:
                    t0 = time.time()
                    try:
                        if model.name == "climatology":
                            # climatologia opera em y nao-escalado e usa mes-alvo
                            model.fit(Xtr_s, ytr, Xva_s, yva, months_train=months_train_target)
                            yhat = model.predict(Xts_s, months_target=months_test_target)
                        elif model.name in {"persistence", "seasonal_naive"}:
                            # baselines triviais nao precisam de fit/scale
                            model.fit(Xtr, ytr, Xva, yva)
                            yhat = model.predict(Xts)
                        elif model.name == "sarima":
                            model.fit(Xtr, ytr, Xva, yva)
                            yhat = model.predict(Xts)
                        else:
                            model.fit(Xtr_s, ytr, Xva_s, yva)
                            yhat = model.predict(Xts_s)
                    except Exception as exc:
                        console.print(f"  [red]falhou[/red] {model.name} h={h} seed={seed}: {exc}")
                        continue
                    dt = time.time() - t0

                    m = all_metrics(yts, yhat)
                    metric_rows.append({
                        "model": model.name, "horizon": h, "fold": fold.fold, "seed": seed,
                        **m, "elapsed_s": round(dt, 2),
                    })
                    for d, yt, yp in zip(dts, yts, yhat):
                        pred_rows.append({
                            "model": model.name, "horizon": h, "fold": fold.fold, "seed": seed,
                            "date": pd.Timestamp(d), "y_true": float(yt), "y_pred": float(yp),
                        })
                    if cfg.verbose:
                        console.print(f"  {model.name:12s} h={h} fold={fold.fold} seed={seed} "
                                      f"rmse={m['rmse']:.3f} r2={m['r2']:.3f} acc={m['acc']:.3f} "
                                      f"({dt:.1f}s)")

    df_m = pd.DataFrame(metric_rows)
    df_p = pd.DataFrame(pred_rows)
    return df_m, df_p
