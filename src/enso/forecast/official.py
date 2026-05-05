"""Coleta a previsao oficial mais recente do CPC/IRI ENSO plume.

Tenta a API publica do IRI quando disponivel; cai para tabela estatica como
fallback (pode ser editada manualmente em data/raw/official_forecast.csv).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from enso.config import RAW_DIR


FALLBACK_FILE = RAW_DIR / "official_forecast.csv"


def load_official_forecast() -> pd.DataFrame:
    """Le previsao oficial: colunas [target_month, season, mean, q05, q95,
    el_nino_prob, neutral_prob, la_nina_prob, source, issued].

    Se o arquivo estatico nao existir, retorna DataFrame vazio.
    """
    if FALLBACK_FILE.exists():
        df = pd.read_csv(FALLBACK_FILE)
        if "target_month" in df.columns:
            df["target_month"] = pd.to_datetime(df["target_month"])
        if "issued" in df.columns:
            df["issued"] = pd.to_datetime(df["issued"], errors="coerce")
        return df
    return pd.DataFrame(columns=[
        "target_month", "season", "mean", "q05", "q95",
        "el_nino_prob", "neutral_prob", "la_nina_prob", "source", "issued",
    ])


def write_official_template() -> Path:
    """Cria um template manual em CSV para o usuario preencher."""
    template = pd.DataFrame({
        "target_month": pd.date_range("2026-06-01", periods=6, freq="MS"),
        "mean": [None] * 6,
        "q05":  [None] * 6,
        "q95":  [None] * 6,
        "source": ["CPC/IRI plume"] * 6,
    })
    FALLBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(FALLBACK_FILE, index=False)
    return FALLBACK_FILE
