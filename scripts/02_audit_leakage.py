"""Roda auditoria de leakage e EDA. Salva tabelas e checks em runs/audit/."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import MinMaxScaler

from enso.config import RUNS_DIR, TARGET_COL
from enso.data.leakage import (
    assert_no_centered_rolling,
    assert_scaler_fitted_on_train,
    assert_split_no_overlap,
    assert_target_not_in_features,
)
from enso.data.loader import CORE_FEATURES, coverage_report, load_master, select_features
from enso.data.splits import summarize_splits, walk_forward
from enso.features.engineer import make_supervised

console = Console()


def main():
    out_dir = RUNS_DIR / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]Carregamento e cobertura")
    df = load_master(since="1982-01-01")
    cov = coverage_report(df)
    cov.to_csv(out_dir / "coverage.csv")
    console.print(cov.to_string())

    console.rule("[bold cyan]Selecao de features (drop colineares com o alvo)")
    df_sel = select_features(df, target_col=TARGET_COL)
    console.print(f"colunas restantes: {list(df_sel.columns)}")

    feature_cols = [c for c in CORE_FEATURES if c in df_sel.columns and c != TARGET_COL]
    console.print(f"feature_cols ({len(feature_cols)}): {feature_cols}")

    console.rule("[bold cyan]Walk-forward splits")
    # Para inspecao usamos o supervised do horizonte 1
    X, y = make_supervised(
        df_sel, feature_cols=feature_cols, target_col=TARGET_COL, horizon=1
    )
    console.print(f"X: {X.shape}, y: {y.shape}, periodo: {y.index.min().date()} -> {y.index.max().date()}")

    splits = list(
        walk_forward(
            dates=y.index, n_folds=5, val_size=24, test_size=24, min_train_size=180
        )
    )
    summary = summarize_splits(splits)
    summary.to_csv(out_dir / "splits_h1.csv", index=False)
    table = Table(title="Splits walk-forward (horizonte 1)")
    for c in summary.columns:
        table.add_column(c)
    for _, row in summary.iterrows():
        table.add_row(*[str(v) for v in row.values])
    console.print(table)

    console.rule("[bold cyan]Auditoria de leakage")

    console.print("[1] no overlap entre train/val/test...")
    assert_split_no_overlap(splits)
    console.print("    [green]ok[/green]")

    console.print("[2] alvo nao esta nas features...")
    assert_target_not_in_features(list(X.columns), TARGET_COL)
    console.print("    [green]ok[/green]")

    console.print("[3] nenhuma feature tem corr ~= 1 com o alvo no instante t (overlap estrutural)...")
    assert_no_centered_rolling(X, y, max_corr_at_zero_lag=0.999)
    console.print("    [green]ok[/green]")

    console.print("[4] scaler fitado SO no treino (simulacao por fold)...")
    for s in splits:
        Xt = X.values[s.train_idx]
        scaler = MinMaxScaler().fit(Xt)
        assert_scaler_fitted_on_train(scaler.data_min_, Xt)
    console.print("    [green]ok[/green]")

    console.rule("[bold cyan]Sanity: correlacoes do alvo (teto sem leakage)")
    corrs = X.corrwith(y).abs().sort_values(ascending=False)
    top = corrs.head(20)
    console.print(top.to_string())
    top.to_csv(out_dir / "top20_corr_h1.csv")

    console.rule("[bold cyan]Comparacao: ONI vs nino34_anom (mostra o leakage do trabalho anterior)")
    full = load_master(since="1982-01-01")
    if {"oni", "nino34_anom"}.issubset(full.columns):
        oni = full["oni"].dropna()
        nino = full["nino34_anom"].dropna()
        common = oni.index.intersection(nino.index)
        oni = oni.loc[common]
        nino = nino.loc[common]
        # ONI(t) vs ONI(t-1) - corr esperada altissima por overlap
        oni_lag1 = oni.shift(1).dropna()
        oni_t = oni.loc[oni_lag1.index]
        c_oni = float(np.corrcoef(oni_lag1.values, oni_t.values)[0, 1])
        # nino34_anom(t) vs nino34_anom(t-1)
        n_lag1 = nino.shift(1).dropna()
        n_t = nino.loc[n_lag1.index]
        c_nino = float(np.corrcoef(n_lag1.values, n_t.values)[0, 1])
        console.print(f"corr(ONI(t), ONI(t-1))             = {c_oni:.4f}  <- inflado por media movel 3m")
        console.print(f"corr(nino34_anom(t), nino34_anom(t-1)) = {c_nino:.4f}  <- limite real (persistencia)")
        console.print(f"[bold]gap[/bold] = {c_oni - c_nino:+.4f}  (quanto da 'previsao' do trabalho anterior era leakage)")

    console.print(f"\n[green]Auditoria concluida. Saidas em {out_dir}[/green]")


if __name__ == "__main__":
    main()
