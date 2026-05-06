"""Helpers compartilhados pelas paginas do Streamlit."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# adiciona src/ ao path se rodando standalone
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enso.config import FORECAST_DIR, MASTER_PARQUET, RUNS_DIR  # noqa: E402

PHASES = [
    ("La Nina forte",     -99.0, -1.5, "#08306b"),
    ("La Nina moderada",  -1.5,  -1.0, "#2171b5"),
    ("La Nina fraca",     -1.0,  -0.5, "#6baed6"),
    ("Neutro",            -0.5,   0.5, "#d9d9d9"),
    ("El Nino fraco",      0.5,   1.0, "#fcae91"),
    ("El Nino moderado",   1.0,   1.5, "#fb6a4a"),
    ("El Nino forte",      1.5,   2.0, "#cb181d"),
    ("El Nino muito forte",2.0,  99.0, "#67000d"),
]


def phase_label(value: float) -> tuple[str, str]:
    """Retorna (rotulo, hex_color) para o ONI/anom value dado."""
    for name, lo, hi, color in PHASES:
        if lo <= value < hi:
            return name, color
    return "Indefinido", "#999999"


@st.cache_data(show_spinner=False, ttl=600)
def load_master(since: str | None = None, until: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(MASTER_PARQUET)
    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    if since is not None:
        df = df[df.index >= pd.Timestamp(since)]
    if until is not None:
        df = df[df.index <= pd.Timestamp(until)]
    return df


@st.cache_data(show_spinner=False, ttl=600)
def load_forecast() -> pd.DataFrame:
    p = FORECAST_DIR / "forecast_future.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["origin"] = pd.to_datetime(df["origin"])
    df["target"] = pd.to_datetime(df["target"])
    return df


@st.cache_data(show_spinner=False, ttl=600)
def load_metrics(tag: str = "full_v1") -> pd.DataFrame:
    """Carrega metrics: tenta parquet (menor) e cai para csv."""
    base = RUNS_DIR / f"train_{tag}"
    pq = base / "metrics.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = base / "metrics.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=600)
def load_predictions(tag: str = "full_v1") -> pd.DataFrame:
    base = RUNS_DIR / f"train_{tag}"
    pq = base / "predictions.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = base / "predictions.csv"
    if csv.exists():
        return pd.read_csv(csv, parse_dates=["date"])
    return pd.DataFrame()


def latest_observation(df: pd.DataFrame, col: str = "nino34_anom") -> tuple[pd.Timestamp, float]:
    s = df[col].dropna()
    return s.index[-1], float(s.iloc[-1])


def _text_color_for_bg(hex_color: str) -> str:
    """Retorna 'white' ou '#222' conforme luminancia do fundo (WCAG simples)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return "#222" if lum > 0.6 else "white"


def phase_badge(value: float) -> str:
    label, color = phase_label(value)
    txt = _text_color_for_bg(color)
    arrow = "+" if value > 0 else ("-" if value < 0 else "=")
    return (
        f"<div style='display:inline-block; padding:0.5rem 1rem; "
        f"background:{color}; color:{txt}; border-radius:8px; "
        f"font-weight:700; font-size:1.15rem; "
        f"border:1px solid rgba(0,0,0,0.08); box-shadow:0 1px 3px rgba(0,0,0,0.08);'>"
        f"{arrow} {label} ({value:+.2f} degC)</div>"
    )
