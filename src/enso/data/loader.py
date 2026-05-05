"""Carrega dataset mestre e expoe utilitarios de selecao temporal/feature."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from enso.config import MASTER_PARQUET, TARGET_COL


CORE_FEATURES = [
    # Indices de SST (CPC sstoi, desde 1982)
    "nino12_sst", "nino12_anom",
    "nino3_sst",  "nino3_anom",
    "nino4_sst",  "nino4_anom",
    "nino34_sst", "nino34_anom",
    # Indices atmosfericos / oceanicos de superficie
    "soi", "mei", "olr", "pna", "iod", "qbo", "tni", "pdo",
    # Subsuperficie: anomalia de WWV (warm water volume) e T300 (T media 0-300m).
    # Lideram SST equatorial em ~6 meses (Meinen & McPhaden 2000) - pesquisa
    # publicada usa para extender skill alem do limite de superficie.
    "wwv_anom", "t300_anom",
    # ONI (excluido em select_features - media movel trimestral)
    # AMO excluido (parou em 2023 e quebra forecast atual)
    # sunspots: proxy solar
    "sunspots",
]


def fill_recent_gaps(df: pd.DataFrame, max_carry: int = 4) -> pd.DataFrame:
    """Forward-fill limitado para imputar latencia de publicacao em indices NOAA.

    Alguns indices (OLR, TNI) tipicamente atrasam 1-3 meses em relacao aos
    indices de SST. Usar carry-forward limitado evita perder a janela atual.
    Implementacao via pd.ffill(limit=max_carry).
    """
    return df.ffill(limit=max_carry)


def load_master(
    path: Path | None = None,
    since: str | None = "1982-01-01",
    until: str | None = None,
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Carrega o dataset mestre, opcionalmente filtrando data e colunas."""
    p = path or MASTER_PARQUET
    df = pd.read_parquet(p)
    df.index = pd.DatetimeIndex(df.index)
    if since is not None:
        df = df[df.index >= pd.Timestamp(since)]
    if until is not None:
        df = df[df.index <= pd.Timestamp(until)]
    if cols is not None:
        df = df[cols]
    return df.sort_index()


def make_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.Series:
    """Retorna a serie alvo (nino34_anom mensal nao-suavizado por padrao)."""
    if target_col not in df.columns:
        raise KeyError(f"Coluna alvo {target_col!r} nao esta no master.")
    return df[target_col].copy()


def select_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    drop_target_collinear: bool = True,
    extra_drop: list[str] | None = None,
) -> pd.DataFrame:
    """Seleciona features preditoras, removendo colineares com o alvo se pedido.

    Por construcao, removemos:
      - oni: media movel trimestral centrada (overlap estrutural com alvo).
      - nino34_sst e nino34_anom_long: redundantes/colineares com nino34_anom.
      - mei (opcional): contem nino34 anomaly como sub-componente.
    """
    drop = set()
    if drop_target_collinear:
        drop.update({"oni", "nino34_sst", "nino34_anom_long"})
        # nino34_anom eh o proprio alvo (target_col padrao); preservamos no df
        # mas o trainer vai cuidar de nao incluir o alvo entre features
    if extra_drop:
        drop.update(extra_drop)
    keep = [c for c in df.columns if c not in drop]
    return df[keep].copy()


def coverage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Tabela de cobertura (inicio, fim, n_validos) por coluna."""
    rows = []
    for c in df.columns:
        s = df[c].dropna()
        rows.append({
            "col": c,
            "inicio": s.index.min(),
            "fim": s.index.max(),
            "n_validos": len(s),
            "n_faltantes": df[c].isna().sum(),
            "min": s.min() if len(s) else np.nan,
            "max": s.max() if len(s) else np.nan,
        })
    return pd.DataFrame(rows).set_index("col")
