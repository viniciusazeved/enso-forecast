"""Serie historica completa do ENSO desde 1950."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shared.components import aplicar_estilo, header_pagina
from shared.style import FAIXAS_FASE
from utils import load_master

st.set_page_config(page_title="Serie historica - ENSO", page_icon="🌊", layout="wide")
aplicar_estilo()

header_pagina(
    "Serie historica do ENSO",
    "Indice Niño 3.4 (anomalia mensal) e ONI (media trimestral) desde 1950. "
    "As faixas coloridas marcam intensidade.",
)

df = load_master()

st.subheader("1. Visao geral")

c1, c2, c3 = st.columns(3)
with c1:
    yr_min = int(df.index.year.min())
    yr_max = int(df.index.year.max())
    rng = st.slider("Periodo", yr_min, yr_max, (1980, yr_max))
with c2:
    show_oni = st.checkbox("ONI (media trimestral)", value=True)
    show_anom = st.checkbox("Anomalia mensal Niño 3.4", value=True)
with c3:
    show_phases = st.checkbox("Faixas de intensidade", value=True)
    show_long = st.checkbox("Niño 3.4 longa (CPC, desde 1950)", value=False)

mask = (df.index.year >= rng[0]) & (df.index.year <= rng[1])
sub = df.loc[mask]

fig = go.Figure()
if show_phases:
    for nome, lo, hi, cor in FAIXAS_FASE:
        lo_v = max(lo, -3.0); hi_v = min(hi, 3.0)
        if hi_v <= lo_v: continue
        fig.add_hrect(y0=lo_v, y1=hi_v, fillcolor=cor, opacity=0.15, line_width=0)
if show_anom and "nino34_anom" in sub.columns:
    s = sub["nino34_anom"].dropna()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines", name="Niño 3.4 (anomalia mensal)",
        line=dict(color="#1c2a3a", width=1.3),
    ))
if show_oni and "oni" in sub.columns:
    s = sub["oni"].dropna()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines", name="ONI (3m)",
        line=dict(color="#1f4e79", width=2),
    ))
if show_long and "nino34_anom_long" in sub.columns:
    s = sub["nino34_anom_long"].dropna()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines", name="Niño 3.4 (CPC, detrended)",
        line=dict(color="#ff7f0e", width=1.0, dash="dot"),
    ))
fig.update_layout(
    height=480, margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="data", yaxis_title="degC",
    plot_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    font=dict(family="Inter, -apple-system, Segoe UI, Arial, sans-serif"),
)
fig.update_yaxes(range=[-3.0, 3.0], gridcolor="#eeeeee")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.divider()

st.subheader("2. Eventos extremos historicos")
st.markdown(
    "Os 5 meses com maior anomalia positiva e os 5 com maior anomalia negativa "
    "na regiao Niño 3.4 (1982-presente). Eventos com pico acima de +2,5 degC "
    "ocorreram em 1982-83, 1997-98 e 2015-16."
)
if "nino34_anom" in df.columns:
    s = df["nino34_anom"].dropna()
    extremos = pd.DataFrame({
        "El Nino top-5":  s.nlargest(5).round(2),
        "La Nina top-5":  s.nsmallest(5).round(2),
    })
    st.dataframe(extremos, use_container_width=True)

st.divider()

st.subheader("3. Variaveis disponiveis no dataset mestre")
st.markdown(
    "Todas as series sao publicas e baixadas direto da fonte (NOAA PSL, CPC, "
    "NCEI, SILSO). Algumas comecam mais tarde (MEI v2 desde 1979, OLR desde "
    "1974)."
)
cov = []
for c in df.columns:
    s = df[c].dropna()
    cov.append({
        "variavel": c,
        "inicio": s.index.min(),
        "fim":    s.index.max(),
        "n_validos": len(s),
        "min":    round(s.min(), 3) if len(s) else None,
        "max":    round(s.max(), 3) if len(s) else None,
    })
st.dataframe(pd.DataFrame(cov).set_index("variavel"), use_container_width=True)
