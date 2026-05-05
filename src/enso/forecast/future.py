"""Forecasting futuro: re-treina o vencedor por horizonte com TODO o historico
e gera previsoes para os proximos 1-6 meses, com IC do ensemble multi-seed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from sklearn.preprocessing import StandardScaler

from enso.config import FORECAST_DIR, HORIZONS, RUNS_DIR, TARGET_COL
from enso.data.loader import CORE_FEATURES, fill_recent_gaps, load_master, select_features
from enso.eval.compare import load_metrics, winner_per_horizon
from enso.features.engineer import make_sequences
from enso.models.dl import LSTM, MLP, TCN, Mamba, Transformer
from enso.models.baselines import DLinear, Persistence, SARIMA, SeasonalNaive

console = Console()


def _make_model(name: str, n_features: int, lookback: int, horizon: int,
                target_idx: int, device: str, epochs: int = 300):
    name = name.lower()
    if name == "persistence":
        return Persistence(target_col_idx=target_idx)
    if name == "seasonal_naive":
        return SeasonalNaive(target_col_idx=target_idx, horizon=horizon)
    if name == "sarima":
        return SARIMA(target_col_idx=target_idx, horizon=horizon)
    if name == "dlinear":
        return DLinear(lookback=lookback, horizon=horizon, target_col_idx=target_idx,
                       epochs=epochs, device=device)
    if name == "mlp":
        return MLP(lookback=lookback, n_features=n_features, epochs=epochs, device=device)
    if name == "lstm":
        return LSTM(n_features=n_features, epochs=epochs, device=device)
    if name == "tcn":
        return TCN(n_features=n_features, epochs=epochs, device=device)
    if name == "transformer":
        return Transformer(n_features=n_features, epochs=epochs, device=device)
    if name == "mamba":
        return Mamba(n_features=n_features, epochs=epochs, device=device)
    raise ValueError(f"modelo desconhecido: {name}")


@dataclass
class ForecastConfig:
    train_run_dir: Path                 # pasta com metrics.csv pra eleger vencedores
    horizons: tuple[int, ...] = tuple(HORIZONS)
    lookback: int = 18
    seeds: tuple[int, ...] = tuple(range(10))
    epochs: int = 300
    val_size: int = 24                  # holdout para early stopping
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    since: str = "1982-01-01"
    out_dir: Path = FORECAST_DIR


def run_forecast(cfg: ForecastConfig) -> pd.DataFrame:
    console.print(f"[bold]device={cfg.device} | lookback={cfg.lookback} | seeds={len(cfg.seeds)}[/bold]")
    metrics = load_metrics(cfg.train_run_dir)
    winners = winner_per_horizon(metrics).set_index("horizon")["model"].to_dict()
    console.print(f"[bold]Vencedores por horizonte:[/bold] {winners}")

    df = load_master(since=cfg.since)
    df = select_features(df, target_col=TARGET_COL)

    candidates = [c for c in CORE_FEATURES if c in df.columns and c != TARGET_COL]

    # Ultima observacao disponivel do target
    last_target_date = df[TARGET_COL].dropna().index.max()
    console.print(f"ultimo {TARGET_COL}: {last_target_date.date()} = {df[TARGET_COL].loc[last_target_date]:.3f}")

    # Filtra features cuja ultima observacao esta dentro de max_lag_months do alvo
    max_lag_months = 6
    kept, dropped = [], []
    for c in candidates:
        last_c = df[c].dropna().index.max() if df[c].notna().any() else None
        if last_c is None:
            dropped.append((c, "vazia"))
            continue
        gap = (last_target_date.year - last_c.year) * 12 + (last_target_date.month - last_c.month)
        if gap > max_lag_months:
            dropped.append((c, f"gap={gap}m"))
        else:
            kept.append(c)
    if dropped:
        console.print(f"  [yellow]features descartadas (gap > {max_lag_months}m):[/yellow] " +
                      ", ".join(f"{c}({why})" for c, why in dropped))
    feature_cols = [TARGET_COL] + kept
    target_idx = feature_cols.index(TARGET_COL)
    n_features = len(feature_cols)
    console.print(f"  features ativas: {n_features} -> {feature_cols}")

    # Carry-forward das mantidas (max_carry alinhado com max_lag_months)
    df[feature_cols] = fill_recent_gaps(df[feature_cols], max_carry=max_lag_months)

    rows = []
    for h in cfg.horizons:
        winner = winners.get(h)
        if winner is None:
            console.print(f"  [yellow]sem vencedor para h={h}, pulando[/yellow]")
            continue
        console.rule(f"[cyan]horizonte {h}m -> modelo {winner}")

        X_seq, y_seq, dates = make_sequences(
            df, feature_cols=feature_cols, target_col=TARGET_COL,
            horizon=h, lookback=cfg.lookback,
        )
        # Janela de origem para previsao futura: ultima janela de feature_cols
        feats_full = df[feature_cols].values
        if len(feats_full) < cfg.lookback:
            console.print("  [red]features insuficientes[/red]")
            continue
        # janela com ultimas `lookback` observacoes (todas as features tem que estar nao-NaN)
        # buscamos o maior t com janela completa disponivel
        valid_mask = ~np.isnan(feats_full).any(axis=1)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) < cfg.lookback:
            console.print("  [red]janela final com NaN nas features[/red]")
            continue
        last_t = valid_idx[-1]
        if last_t - cfg.lookback + 1 < 0:
            continue
        X_pred = feats_full[last_t - cfg.lookback + 1 : last_t + 1].astype(float)
        origin_date = df.index[last_t]
        target_date = origin_date + pd.DateOffset(months=h)
        console.print(f"  origem t={origin_date.date()} -> alvo t+h={target_date.date()}")

        # Split treino/val: ultimas val_size amostras = val
        if len(X_seq) <= cfg.val_size + 12:
            console.print("  [yellow]serie curta para val; usando 80/20[/yellow]")
            n_val = max(12, int(len(X_seq) * 0.2))
        else:
            n_val = cfg.val_size
        Xtr, ytr = X_seq[:-n_val], y_seq[:-n_val]
        Xva, yva = X_seq[-n_val:], y_seq[-n_val:]
        # scaler no treino
        F = X_seq.shape[2]
        sc = StandardScaler().fit(Xtr.reshape(-1, F))
        def _t(x): return sc.transform(x.reshape(-1, F)).reshape(x.shape)
        Xtr_s = _t(Xtr); Xva_s = _t(Xva)
        Xpred_s = sc.transform(X_pred.reshape(-1, F)).reshape(1, cfg.lookback, F)

        # Ensemble multi-seed
        preds = []
        for seed in cfg.seeds:
            np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            mdl = _make_model(winner, n_features, cfg.lookback, h, target_idx,
                              device=cfg.device, epochs=cfg.epochs)
            t0 = time.time()
            try:
                if winner in {"persistence", "seasonal_naive", "sarima"}:
                    mdl.fit(Xtr, ytr)
                    p = float(mdl.predict(X_pred[None, ...])[0])
                else:
                    mdl.fit(Xtr_s, ytr, Xva_s, yva)
                    p = float(mdl.predict(Xpred_s)[0])
            except Exception as exc:
                console.print(f"  [red]seed {seed} falhou: {exc}[/red]")
                continue
            preds.append(p)
            console.print(f"  seed={seed} pred={p:+.3f} ({time.time()-t0:.1f}s)")

        if not preds:
            continue
        arr = np.array(preds)
        rows.append({
            "horizon": h,
            "model": winner,
            "origin": origin_date,
            "target": target_date,
            "mean": float(arr.mean()),
            "std":  float(arr.std()),
            "q05":  float(np.quantile(arr, 0.05)),
            "q25":  float(np.quantile(arr, 0.25)),
            "q50":  float(np.quantile(arr, 0.50)),
            "q75":  float(np.quantile(arr, 0.75)),
            "q95":  float(np.quantile(arr, 0.95)),
            "n_seeds": len(arr),
        })

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_fc = pd.DataFrame(rows)
    df_fc.to_csv(out_dir / "forecast_future.csv", index=False)
    df_fc.to_parquet(out_dir / "forecast_future.parquet")
    console.print(df_fc.to_string())
    return df_fc
