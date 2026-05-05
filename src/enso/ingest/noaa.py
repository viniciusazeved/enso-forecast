"""Ingestao de indices climaticos publicos da NOAA PSL, CPC e SILSO.

Cada funcao fetch_* baixa, parseia e retorna pd.Series mensal com DatetimeIndex
no primeiro dia do mes. build_master() consolida tudo num DataFrame mestre.

Fontes usam dois formatos principais:
- Formato PSL: ano + 12 colunas (Jan-Dez), ASCII com missing -99.99 ou similar.
- Formato CPC indices: tabela texto com YR MON e variaveis em colunas.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from rich.console import Console

from enso.config import RAW_DIR, ensure_dirs

console = Console()

USER_AGENT = "enso-forecast/0.1 (+contato@azevedoambiental.com)"
TIMEOUT = 60


SOURCES: dict[str, str] = {
    # ONI: Oceanic Nino Index (media movel trimestral de SST anom Nino 3.4)
    "oni":   "https://psl.noaa.gov/data/correlation/oni.data",
    # Nino regions SST e anomalias (mensal, CPC, desde 1982)
    "sstoi": "https://www.cpc.ncep.noaa.gov/data/indices/sstoi.indices",
    # Nino 3.4 anomaly detrended (CPC, desde 1950) - extende historico antes de 1982
    "nino34_long": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt",
    # SOI: Southern Oscillation Index
    "soi":   "https://psl.noaa.gov/data/correlation/soi.data",
    # MEI v2: Multivariate ENSO Index
    "mei":   "https://psl.noaa.gov/enso/mei/data/meiv2.data",
    # OLR: Outgoing Longwave Radiation (proxy conveccao tropical)
    "olr":   "https://psl.noaa.gov/data/correlation/olr.data",
    # PNA: Pacific North American pattern
    "pna":   "https://psl.noaa.gov/data/correlation/pna.data",
    # PDO: Pacific Decadal Oscillation (NCEI ERSSTv5, desde 1854)
    "pdo":   "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat",
    # AMO: Atlantic Multidecadal Oscillation (unsmoothed)
    "amo":   "https://psl.noaa.gov/data/correlation/amon.us.data",
    # IOD (DMI): Indian Ocean Dipole - Dipole Mode Index
    "iod":   "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data",
    # QBO: Quasi-Biennial Oscillation (vento zonal 30 hPa)
    "qbo":   "https://psl.noaa.gov/data/correlation/qbo.data",
    # TNI: Trans-Nino Index (gradient SST 1+2 -> 4)
    "tni":   "https://psl.noaa.gov/data/correlation/tni.data",
    # SILSO: numero mensal de manchas solares (proxy atividade solar)
    "sunspots": "https://www.sidc.be/SILSO/INFO/snmtotcsv.php",
}


def _download(name: str, url: str, force: bool = False) -> Path:
    """Baixa o arquivo bruto e salva em data/raw/{name}.{ext}."""
    ensure_dirs()
    ext = ".csv" if name == "sunspots" else ".dat" if name == "pdo" else ".txt"
    out = RAW_DIR / f"{name}{ext}"
    if out.exists() and not force:
        return out
    console.print(f"[cyan]baixando[/cyan] {name} <- {url}")
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out


def _parse_year_columns(
    path: Path, missing: float = 99.99, missing_threshold: float = 90.0
) -> pd.Series:
    """Parser para formato 'year + 12 colunas' com header descritivo livre.

    Diferente de _parse_psl_format, aqui nao ha header com (y0, y1). O parser:
      1. ignora qualquer linha cujo primeiro token nao seja int de 4 digitos,
      2. monta serie a partir de linhas com (year, m1..m12).
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    rows: list[tuple[int, list[float]]] = []
    for ln in text.splitlines():
        parts = ln.split()
        if len(parts) != 13:
            continue
        if not re.fullmatch(r"\d{4}", parts[0]):
            continue
        try:
            year = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        rows.append((year, vals))
    if not rows:
        raise ValueError(f"Nenhuma linha de dados em {path.name}")
    records = []
    for year, vals in rows:
        for m, v in enumerate(vals, start=1):
            records.append((pd.Timestamp(year=year, month=m, day=1), v))
    s = pd.Series(
        [r[1] for r in records],
        index=pd.DatetimeIndex([r[0] for r in records], name="date"),
        name=path.stem,
    )
    s = s.where(s.abs() < missing_threshold)
    return s.sort_index()


def _parse_psl_format(path: Path, missing_threshold: float = -90.0) -> pd.Series:
    """Parser do formato NOAA PSL: ano + 12 colunas (Jan-Dez).

    Estrutura:
        Linha 1: '<year_start> <year_end>'
        Linhas 2..N: '<year>  m1  m2 ... m12'
        Apos os dados: linha com codigo de missing (ex.: -99.9), depois texto.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    header = lines[0].split()
    if len(header) < 2 or not all(re.fullmatch(r"-?\d+", h) for h in header[:2]):
        raise ValueError(f"Header inesperado em {path.name}: {lines[0]!r}")
    y0, y1 = int(header[0]), int(header[1])

    rows: list[tuple[int, list[float]]] = []
    for ln in lines[1:]:
        parts = ln.split()
        if len(parts) != 13:
            continue
        try:
            year = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if year < y0 or year > y1:
            continue
        rows.append((year, vals))

    if not rows:
        raise ValueError(f"Nenhuma linha de dados parseada em {path.name}")

    records = []
    for year, vals in rows:
        for m, v in enumerate(vals, start=1):
            records.append((pd.Timestamp(year=year, month=m, day=1), v))
    s = pd.Series(
        data=[r[1] for r in records],
        index=pd.DatetimeIndex([r[0] for r in records], name="date"),
        name=path.stem,
    )
    s = s.where(s > missing_threshold)
    return s.sort_index()


def fetch_oni(force: bool = False) -> pd.Series:
    return _parse_psl_format(_download("oni", SOURCES["oni"], force)).rename("oni")


def fetch_soi(force: bool = False) -> pd.Series:
    return _parse_psl_format(_download("soi", SOURCES["soi"], force)).rename("soi")


def fetch_olr(force: bool = False) -> pd.Series:
    return _parse_psl_format(_download("olr", SOURCES["olr"], force)).rename("olr")


def fetch_pna(force: bool = False) -> pd.Series:
    return _parse_psl_format(_download("pna", SOURCES["pna"], force)).rename("pna")


def fetch_amo(force: bool = False) -> pd.Series:
    return _parse_psl_format(_download("amo", SOURCES["amo"], force)).rename("amo")


def fetch_iod(force: bool = False) -> pd.Series:
    return _parse_psl_format(_download("iod", SOURCES["iod"], force)).rename("iod")


def fetch_qbo(force: bool = False) -> pd.Series:
    return _parse_psl_format(_download("qbo", SOURCES["qbo"], force)).rename("qbo")


def fetch_tni(force: bool = False) -> pd.Series:
    return _parse_psl_format(_download("tni", SOURCES["tni"], force)).rename("tni")


def fetch_mei(force: bool = False) -> pd.Series:
    """MEI v2: formato similar ao PSL mas comeca em 1979 e tem header descritivo."""
    path = _download("mei", SOURCES["mei"], force)
    return _parse_psl_format(path).rename("mei")


def fetch_sstoi(force: bool = False) -> pd.DataFrame:
    """sstoi.indices do CPC: SST e anomalias para Nino 1+2, 3, 4 e 3.4.

    Layout esperado (espacos variaveis):
        YR  MON  NINO1+2  ANOM  NINO3  ANOM  NINO4  ANOM  NINO3.4  ANOM

    O header tem 'ANOM' duplicado; pandas adiciona sufixos. Parseamos manualmente.
    """
    path = _download("sstoi", SOURCES["sstoi"], force)
    columns = [
        "yr", "mon",
        "nino12_sst", "nino12_anom",
        "nino3_sst",  "nino3_anom",
        "nino4_sst",  "nino4_anom",
        "nino34_sst", "nino34_anom",
    ]
    df = pd.read_csv(
        path, sep=r"\s+", engine="python", skiprows=1, header=None, names=columns,
    )
    df["date"] = pd.to_datetime(
        df["yr"].astype(int).astype(str) + "-" + df["mon"].astype(int).astype(str) + "-01"
    )
    df = df.drop(columns=["yr", "mon"]).set_index("date").sort_index()
    return df.astype(float)


def fetch_nino34_long(force: bool = False) -> pd.Series:
    """Anomalia detrended de SST Nino 3.4 mensal (CPC, desde 1950).

    Layout (espacos variaveis):
        YR  MON  TOTAL  ClimAdjust  ANOM
    Estende a historia antes de 1982 (limite do sstoi.indices).
    """
    path = _download("nino34_long", SOURCES["nino34_long"], force)
    df = pd.read_csv(
        path, sep=r"\s+", engine="python", skiprows=1, header=None,
        names=["yr", "mon", "total", "climadj", "anom"],
    )
    df["date"] = pd.to_datetime(
        df["yr"].astype(int).astype(str) + "-" + df["mon"].astype(int).astype(str) + "-01"
    )
    return df.set_index("date")["anom"].astype(float).rename("nino34_anom_long").sort_index()


def fetch_pdo(force: bool = False) -> pd.Series:
    """PDO do NCEI ERSSTv5 (desde 1854).

    Layout: 'Year + 12 colunas (Jan-Dez)' com header descritivo nas primeiras
    duas linhas. Missing flag = 99.99.
    """
    path = _download("pdo", SOURCES["pdo"], force)
    return _parse_year_columns(path).rename("pdo")


def fetch_sunspots(force: bool = False) -> pd.Series:
    """SILSO: numero medio mensal de manchas solares.

    Formato CSV com ';': year; month; decimal_year; sn_mean; sn_std; n_obs; provisional
    Missing flag: -1 em sn_mean.
    """
    path = _download("sunspots", SOURCES["sunspots"], force)
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["year", "month", "decimal_year", "sn_mean", "sn_std", "n_obs", "provisional"],
        engine="python",
    )
    df["date"] = pd.to_datetime(
        df["year"].astype(int).astype(str) + "-" + df["month"].astype(int).astype(str) + "-01"
    )
    s = df.set_index("date")["sn_mean"].rename("sunspots").sort_index()
    return s.where(s >= 0)


def build_master(force: bool = False, since: str = "1950-01-01") -> pd.DataFrame:
    """Consolida todas as fontes num DataFrame mensal alinhado por data.

    Retorna df com colunas:
        nino12_sst, nino12_anom, nino3_sst, nino3_anom, nino4_sst, nino4_anom,
        nino34_sst, nino34_anom, oni, soi, mei, olr, pna, amo, iod, qbo, tni,
        pdo, sunspots
    """
    parts: list[pd.Series | pd.DataFrame] = []

    fetchers: list[tuple[str, callable]] = [
        ("sstoi",        fetch_sstoi),
        ("nino34_long",  fetch_nino34_long),
        ("oni",          fetch_oni),
        ("soi",          fetch_soi),
        ("mei",          fetch_mei),
        ("olr",          fetch_olr),
        ("pna",          fetch_pna),
        ("amo",          fetch_amo),
        ("iod",          fetch_iod),
        ("qbo",          fetch_qbo),
        ("tni",          fetch_tni),
        ("pdo",          fetch_pdo),
        ("sunspots",     fetch_sunspots),
    ]
    for name, fn in fetchers:
        try:
            obj = fn(force)
            parts.append(obj)
            n = len(obj)
            console.print(f"  [green]ok[/green] {name:8s} {n:5d} pontos")
        except Exception as exc:
            console.print(f"  [red]falhou[/red] {name}: {exc}")

    df = pd.concat(parts, axis=1)
    # Garante index mensal monotonico
    df = df.sort_index()
    df = df[df.index >= pd.Timestamp(since)]
    df.index.name = "date"
    return df


def save_master(df: pd.DataFrame) -> None:
    from enso.config import MASTER_CSV, MASTER_PARQUET, ensure_dirs
    ensure_dirs()
    df.to_csv(MASTER_CSV)
    df.to_parquet(MASTER_PARQUET)
    console.print(
        f"[green]salvo[/green] {MASTER_CSV.relative_to(MASTER_CSV.parents[2])} "
        f"({df.shape[0]} linhas x {df.shape[1]} colunas)"
    )
