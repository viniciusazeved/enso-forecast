"""Recupera metrics.csv a partir do log de stdout do trainer interrompido.

Util quando o training foi abortado antes de salvar os arquivos finais.
Parseia linhas do tipo:
    persistence  h=1 fold=0 seed=0 rmse=0.275 r2=0.754 acc=0.874 (0.0s)

Output: runs/train_<tag>/metrics.csv com colunas
[model, horizon, fold, seed, rmse, r2, acc, elapsed_s]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

PAT = re.compile(
    r"(\w[\w-]*)\s+h=(\d+)\s+fold=(\d+)\s+seed=(\d+)\s+"
    r"rmse=([\d\.\-]+)\s+r2=([\d\.\-]+)\s+acc=([\d\.\-]+)\s+\(([\d\.]+)s\)"
)


@app.command()
def main(
    log:    str = typer.Option("runs/train_full_v1.log", help="Arquivo de log."),
    tag:    str = typer.Option("full_v1", help="Tag do run para salvar metrics.csv."),
    drop_partial_horizons: bool = typer.Option(
        True, help="Mantem so horizontes com 5 folds completos."
    ),
):
    p = Path(log)
    if not p.exists():
        console.print(f"[red]Log nao encontrado: {p}[/red]")
        raise typer.Exit(1)

    rows = []
    text = p.read_text(encoding="utf-8", errors="ignore")
    # remove escape sequences ANSI
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    for ln in text.splitlines():
        m = PAT.search(ln)
        if not m:
            continue
        rows.append({
            "model":   m.group(1),
            "horizon": int(m.group(2)),
            "fold":    int(m.group(3)),
            "seed":    int(m.group(4)),
            "rmse":    float(m.group(5)),
            "r2":      float(m.group(6)),
            "acc":     float(m.group(7)),
            "elapsed_s": float(m.group(8)),
        })
    if not rows:
        console.print("[red]Nenhuma linha de metrica encontrada.[/red]")
        raise typer.Exit(1)

    df = pd.DataFrame(rows)
    # MAE e hit nao sao logados; vamos imputar a partir do RMSE como aproximacao grosseira
    # melhor deixar NaN e os consumidores tratam:
    df["mae"] = float("nan")
    df["hit"] = float("nan")

    if drop_partial_horizons:
        # mantem apenas (model, horizon) com 5 folds * 10 seeds = 50 entradas
        counts = df.groupby(["model", "horizon"]).size().reset_index(name="n")
        valid = counts[counts["n"] >= 50][["model", "horizon"]]
        before = len(df)
        df = df.merge(valid, on=["model", "horizon"])
        console.print(
            f"filtro horizontes completos: {before} -> {len(df)} linhas. "
            f"horizontes mantidos: {sorted(df['horizon'].unique().tolist())}"
        )

    out_dir = Path("runs") / f"train_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "metrics.csv"
    df.to_csv(out, index=False)
    console.print(f"[green]Salvo:[/green] {out} ({len(df)} linhas)")

    table = Table(title="Resumo (media por modelo x horizonte)")
    for c in ["model", "horizon", "rmse_mean", "r2_mean", "acc_mean"]:
        table.add_column(c)
    agg = (
        df.groupby(["model", "horizon"])[["rmse", "r2", "acc"]]
        .mean().round(3).reset_index()
    )
    for _, row in agg.iterrows():
        table.add_row(row["model"], str(row["horizon"]),
                      f"{row['rmse']:.3f}", f"{row['r2']:.3f}", f"{row['acc']:.3f}")
    console.print(table)


if __name__ == "__main__":
    app()
