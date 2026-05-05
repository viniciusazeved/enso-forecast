"""Caminhos e constantes centralizadas."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FORECAST_DIR = DATA_DIR / "forecasts"

RUNS_DIR = ROOT / "runs"
CONFIGS_DIR = ROOT / "configs"

MASTER_CSV = PROCESSED_DIR / "enso_master.csv"
MASTER_PARQUET = PROCESSED_DIR / "enso_master.parquet"

TARGET_COL = "nino34_anom"  # anomalia mensal de SST Nino 3.4 (sem media movel)

HORIZONS = [1, 2, 3, 4, 5, 6]
N_SEEDS = 10
DEFAULT_SEED = 42

ENSO_PHASES = {
    "el_nino_muito_forte": 2.0,
    "el_nino_forte": 1.5,
    "el_nino_moderado": 1.0,
    "el_nino_fraco": 0.5,
    "neutro_pos": 0.0,
    "neutro_neg": -0.5,
    "la_nina_fraca": -1.0,
    "la_nina_moderada": -1.5,
    "la_nina_forte": -2.0,
}


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, FORECAST_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
