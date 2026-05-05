"""Mescla resultados de dois runs em uma unica tag (uniao por modelo x horizonte)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import typer
from rich.console import Console

from enso.config import RUNS_DIR

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    src1: str = typer.Argument(..., help="Tag fonte 1"),
    src2: str = typer.Argument(..., help="Tag fonte 2"),
    dst:  str = typer.Argument(..., help="Tag destino"),
):
    p1 = RUNS_DIR / f"train_{src1}"
    p2 = RUNS_DIR / f"train_{src2}"
    pd_dst = RUNS_DIR / f"train_{dst}"
    pd_dst.mkdir(parents=True, exist_ok=True)

    # metrics
    m1 = pd.read_csv(p1 / "metrics.csv") if (p1 / "metrics.csv").exists() else pd.DataFrame()
    m2 = pd.read_csv(p2 / "metrics.csv") if (p2 / "metrics.csv").exists() else pd.DataFrame()
    metrics = pd.concat([m1, m2], ignore_index=True).drop_duplicates(
        subset=["model", "horizon", "fold", "seed"], keep="last",
    )
    metrics.to_csv(pd_dst / "metrics.csv", index=False)
    console.print(f"[green]metrics:[/green] {len(m1)} + {len(m2)} -> {len(metrics)} linhas")
    console.print(f"horizontes em destino: {sorted(metrics['horizon'].unique().tolist())}")

    # predictions
    p1p = p1 / "predictions.csv"
    p2p = p2 / "predictions.csv"
    if p1p.exists() and p2p.exists():
        pp1 = pd.read_csv(p1p, parse_dates=["date"])
        pp2 = pd.read_csv(p2p, parse_dates=["date"])
        preds = pd.concat([pp1, pp2], ignore_index=True).drop_duplicates(
            subset=["model", "horizon", "fold", "seed", "date"], keep="last",
        )
        preds.to_csv(pd_dst / "predictions.csv", index=False)
        console.print(f"[green]predictions:[/green] {len(pp1)} + {len(pp2)} -> {len(preds)} linhas")
    elif p1p.exists() or p2p.exists():
        only = p1p if p1p.exists() else p2p
        pd.read_csv(only, parse_dates=["date"]).to_csv(pd_dst / "predictions.csv", index=False)
        console.print(f"[yellow]predictions:[/yellow] so um lado tinha; copiado de {only.parent.name}")

    console.print(f"[bold green]salvo em:[/bold green] {pd_dst}")


if __name__ == "__main__":
    app()
