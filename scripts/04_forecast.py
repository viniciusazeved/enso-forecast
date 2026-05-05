"""Re-treina o vencedor por horizonte com todo o historico e gera previsao 1-6m."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from rich.console import Console

from enso.config import FORECAST_DIR, RUNS_DIR
from enso.forecast.future import ForecastConfig, run_forecast

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    train_tag: str = typer.Option("full_v1", help="Tag do run de training para eleger vencedores."),
    horizons:  str = typer.Option("1,2,3,4,5,6"),
    lookback:  int = typer.Option(18),
    seeds:     str = typer.Option("0,1,2,3,4,5,6,7,8,9"),
    epochs:    int = typer.Option(300),
):
    cfg = ForecastConfig(
        train_run_dir=RUNS_DIR / f"train_{train_tag}",
        horizons=tuple(int(h) for h in horizons.split(",")),
        lookback=lookback,
        seeds=tuple(int(s) for s in seeds.split(",")),
        epochs=epochs,
    )
    df_fc = run_forecast(cfg)
    console.rule("[green]Forecast salvo")
    console.print(df_fc.to_string())


if __name__ == "__main__":
    app()
