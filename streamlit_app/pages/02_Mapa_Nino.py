"""Mapa interativo das regioes Niño no Pacifico equatorial."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import plotly.graph_objects as go
import streamlit as st

from shared.components import aplicar_estilo, header_pagina, kpi_card
from utils import load_master

st.set_page_config(page_title="Mapa Niño - ENSO", page_icon="🌊", layout="wide")
aplicar_estilo()

header_pagina(
    "Regioes Niño no Pacifico Equatorial",
    "As 4 caixas oceanicas que a NOAA monitora para diagnosticar o ENSO. "
    "A Niño 3.4 e a referencia oficial.",
)

st.subheader("1. Onde fica cada regiao")

REGIOES = {
    "Niño 1+2": {"lon": (-90.0, -80.0), "lat": (-10.0, 0.0), "color": "#d62728",
                 "desc": "Costa do Peru e Equador. Estreita faixa proxima a costa, "
                         "muito sensivel a ondas Kelvin que se propagam pelo Pacifico."},
    "Niño 3":   {"lon": (-150.0, -90.0), "lat": (-5.0, 5.0), "color": "#ff7f0e",
                 "desc": "Pacifico tropical leste. Captura o El Nino classico, com "
                         "aquecimento concentrado no leste."},
    "Niño 3.4": {"lon": (-170.0, -120.0), "lat": (-5.0, 5.0), "color": "#1f4e79",
                 "desc": "Pacifico central. **Referencia oficial**. O ONI e a "
                         "media movel de 3 meses da anomalia desta caixa."},
    "Niño 4":   {"lon": (-160.0, -150.0), "lat": (-5.0, 5.0), "color": "#2ca02c",
                 "desc": "Pacifico central-oeste. Captura padrao Modoki - El Nino "
                         "com aquecimento mais a oeste."},
}

df = load_master()

fig = go.Figure()
for nome, info in REGIOES.items():
    lon0, lon1 = info["lon"]; lat0, lat1 = info["lat"]
    fig.add_trace(go.Scattergeo(
        lon=[lon0, lon1, lon1, lon0, lon0],
        lat=[lat0, lat0, lat1, lat1, lat0],
        fill="toself",
        fillcolor=info["color"],
        opacity=0.45,
        line=dict(color=info["color"], width=2),
        name=nome,
        hovertext=info["desc"],
        hoverinfo="text+name",
    ))
fig.update_geos(
    projection_type="natural earth",
    showcountries=True, countrycolor="rgba(0,0,0,0.2)",
    showland=True, landcolor="#f5f5f5",
    showocean=True, oceancolor="#e8f1fa",
    lataxis_range=[-30, 30],
    lonaxis_range=[-200, 0],
)
fig.update_layout(
    height=480, margin=dict(l=10, r=10, t=10, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    font=dict(family="Inter, -apple-system, Segoe UI, Arial, sans-serif"),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.divider()

st.subheader("2. Anomalia atual em cada regiao")
st.caption(f"Ultima leitura disponivel: {df['nino34_anom'].dropna().index[-1]:%b/%Y}")

cols = ["nino12_anom", "nino3_anom", "nino34_anom", "nino4_anom"]
labels = ["Niño 1+2", "Niño 3", "Niño 3.4", "Niño 4"]
metrics = st.columns(len(cols))
for c, label, key in zip(metrics, labels, cols):
    if key in df.columns:
        s = df[key].dropna()
        if len(s):
            v = float(s.iloc[-1])
            tone = "warn" if v > 1.0 else ("good" if v < -0.5 else "neutral")
            with c:
                kpi_card(label, f"{v:+.2f} degC", tone=tone)

st.divider()

st.subheader("3. Por que a Niño 3.4 e a regiao oficial")
st.markdown(
    """
A Niño 3.4 (170 W a 120 W, +/-5 graus de latitude) fica no **centro de massa**
do aquecimento/resfriamento do Pacifico equatorial durante o ENSO. Ela tem
duas vantagens:

1. **Cobre o territorio onde o ENSO se manifesta com mais clareza** - tanto
   o componente leste (regiao Niño 3, El Nino classico) quanto o componente
   central (regiao Niño 4, El Nino Modoki) acontecem nas vizinhancas dela.
2. **E historica e operacionalmente estavel** desde a definicao do ONI
   (Oceanic Niño Index) pela NOAA. Permite comparar diretamente os eventos
   ao longo de decadas.

A definicao formal de evento ENSO usa o **ONI** (media movel trimestral da
anomalia de SST na Niño 3.4):

- **El Nino**: ONI maior ou igual a +0,5 degC por **5 trimestres consecutivos**.
- **La Nina**: ONI menor ou igual a -0,5 degC por 5 trimestres consecutivos.

Picos isolados nao caracterizam evento. Por isso o ENSO e descrito sempre
como **fase prolongada**, nao como uma medicao pontual.
"""
)
