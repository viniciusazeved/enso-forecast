"""Treina baselines + DL multi-horizonte com multi-seed e walk-forward CV."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from rich.console import Console

from enso.config import RUNS_DIR
from enso.train.trainer import TrainConfig, run

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    horizons: str = typer.Option("1,2,3,4,5,6", help="Lista CSV de horizontes."),
    seeds:    str = typer.Option("0,1,2,3,4,5,6,7,8,9", help="Lista CSV de seeds."),
    folds:    int = typer.Option(5, help="Numero de folds walk-forward."),
    epochs:   int = typer.Option(200, help="Epocas maximas DL."),
    lookback: int = typer.Option(12, help="Janela de lookback (meses)."),
    models:   str = typer.Option(
        "persistence,seasonal_naive,climatology,sarima,dlinear,mlp,lstm,tcn,transformer,mamba",
        help="Modelos a incluir (CSV).",
    ),
    quick:    bool = typer.Option(False, "--quick", help="Modo rapido: 2 seeds, 50 epocas."),
    verbose:  bool = typer.Option(False, "--verbose"),
    tag:      str  = typer.Option("v1", help="Tag para nomear pasta de saida."),
):
    cfg = TrainConfig(
        horizons=tuple(int(h) for h in horizons.split(",")),
        seeds=tuple(int(s) for s in seeds.split(",")),
        n_folds=folds,
        epochs=epochs,
        lookback=lookback,
        include_models=tuple(m.strip() for m in models.split(",")),
        verbose=verbose,
    )
    if quick:
        cfg.seeds = (0, 1)
        cfg.epochs = 50

    out_dir = RUNS_DIR / f"train_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Saida:[/bold] {out_dir}")
    df_m, df_p = run(cfg)

    df_m.to_csv(out_dir / "metrics.csv", index=False)
    df_p.to_csv(out_dir / "predictions.csv", index=False)

    console.rule("[bold green]Resumo (media por modelo x horizonte)")
    pivot = (
        df_m.groupby(["model", "horizon"])[["mae", "rmse", "r2", "acc", "hit"]]
        .agg(["mean", "std"])
        .round(3)
    )
    console.print(pivot.to_string())
    pivot.to_csv(out_dir / "summary_meanstd.csv")


if __name__ == "__main__":
    app()
