"""Baixa, parseia e consolida indices climaticos da NOAA/CPC/SILSO."""
from __future__ import annotations

import sys
from pathlib import Path

# adiciona src/ ao path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from rich.console import Console
from rich.table import Table

from enso.ingest.noaa import build_master, save_master

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    force: bool = typer.Option(False, "--force", help="Re-baixa mesmo se cache local existir."),
    since: str = typer.Option("1950-01-01", "--since", help="Data inicial do mestre."),
):
    """Constroi o dataset mestre."""
    console.rule("[bold cyan]Ingestao NOAA + CPC + SILSO")
    df = build_master(force=force, since=since)
    save_master(df)

    console.rule("[bold]Resumo")
    table = Table(show_header=True, header_style="bold")
    table.add_column("variavel")
    table.add_column("inicio", justify="right")
    table.add_column("fim", justify="right")
    table.add_column("n_validos", justify="right")
    table.add_column("min", justify="right")
    table.add_column("max", justify="right")
    for c in df.columns:
        s = df[c].dropna()
        if len(s) == 0:
            table.add_row(c, "-", "-", "0", "-", "-")
        else:
            table.add_row(
                c,
                str(s.index.min().date()),
                str(s.index.max().date()),
                f"{len(s)}",
                f"{s.min():.3f}",
                f"{s.max():.3f}",
            )
    console.print(table)


if __name__ == "__main__":
    app()
