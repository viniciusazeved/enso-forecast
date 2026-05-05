"""Forecast 1-6 meses do nosso ensemble + comparacao com IRI/CPC."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from enso.forecast.official import load_official_forecast  # noqa: E402

from shared.components import aplicar_estilo, header_pagina
from shared.style import FAIXAS_FASE, fase_para_value
from utils import load_forecast, load_master

st.set_page_config(page_title="Forecast - ENSO", page_icon="🌊", layout="wide")
aplicar_estilo()

header_pagina(
    "Forecast 1-6 meses",
    "Previsao do nosso ensemble multi-seed lado a lado com a previsao oficial "
    "do CPC/IRI. Concordancias e divergencias ficam visiveis.",
)

df = load_master()
fc = load_forecast()
off = load_official_forecast()

if fc.empty:
    st.warning(
        "Forecast futuro ainda nao gerado. Rode `scripts/04_forecast.py` apos "
        "o training completar."
    )
    st.stop()

last_date = df["nino34_anom"].dropna().index[-1]
last_val  = float(df["nino34_anom"].dropna().iloc[-1])

# 1. Tabela
st.subheader("1. Tabela do forecast")
st.markdown(
    "Cada linha mostra um horizonte (1 a 6 meses a frente da ultima observacao). "
    "O modelo vencedor por horizonte foi eleito por score unificado (ver pagina **Modelos**)."
)

show = fc.copy()
show["target"] = show["target"].dt.strftime("%b/%Y")
show["fase prevista"] = fc["mean"].apply(lambda v: fase_para_value(v)[0])
st.dataframe(
    show[["horizon", "target", "model", "mean", "q05", "q25", "q75", "q95",
          "n_seeds", "fase prevista"]].rename(columns={
        "horizon": "h (meses)",
        "target":  "alvo",
        "model":   "modelo",
        "mean":    "media",
        "n_seeds": "seeds",
    }),
    use_container_width=True, hide_index=True,
)

st.divider()

# 2. Visualizacao
st.subheader("2. Visualizacao com intervalo de confianca")
serie = df["nino34_anom"].dropna()
hist_window = st.slider("Janela historica (anos)", 1, 20, 6)
serie = serie[serie.index >= serie.index.max() - pd.DateOffset(years=hist_window)]

fig = go.Figure()
for nome, lo, hi, cor in FAIXAS_FASE:
    lo_v = max(lo, -3.0); hi_v = min(hi, 3.0)
    if hi_v <= lo_v: continue
    fig.add_hrect(y0=lo_v, y1=hi_v, fillcolor=cor, opacity=0.16, line_width=0)
fig.add_trace(go.Scatter(
    x=serie.index, y=serie.values, mode="lines", name="observado",
    line=dict(color="#1c2a3a", width=2),
))

fc_sorted = fc.sort_values("horizon")
ref = pd.DataFrame({
    "date": [last_date] + fc_sorted["target"].tolist(),
    "mean": [last_val] + fc_sorted["mean"].tolist(),
    "q05":  [last_val] + fc_sorted["q05"].tolist(),
    "q95":  [last_val] + fc_sorted["q95"].tolist(),
})
fig.add_trace(go.Scatter(
    x=ref["date"], y=ref["mean"], mode="lines+markers",
    name="nosso ensemble",
    line=dict(color="#1f4e79", width=2.5, dash="dash"),
    marker=dict(size=10, symbol="diamond"),
))
fig.add_trace(go.Scatter(
    x=list(ref["date"]) + list(ref["date"][::-1]),
    y=list(ref["q95"])  + list(ref["q05"][::-1]),
    fill="toself", fillcolor="rgba(31,78,121,0.20)",
    line=dict(color="rgba(0,0,0,0)"), name="IC nosso 90%", hoverinfo="skip",
))

if not off.empty and "mean" in off.columns and off["mean"].notna().any():
    off_sorted = off.dropna(subset=["mean"]).sort_values("target_month")
    fig.add_trace(go.Scatter(
        x=off_sorted["target_month"], y=off_sorted["mean"],
        mode="lines+markers", name="oficial IRI/CPC",
        line=dict(color="#d62728", width=2.5, dash="dot"),
        marker=dict(size=10, symbol="x"),
    ))
    if {"q05", "q95"}.issubset(off.columns):
        fig.add_trace(go.Scatter(
            x=list(off_sorted["target_month"]) + list(off_sorted["target_month"][::-1]),
            y=list(off_sorted["q95"]) + list(off_sorted["q05"][::-1]),
            fill="toself", fillcolor="rgba(214,39,40,0.16)",
            line=dict(color="rgba(0,0,0,0)"), name="IC oficial 90%", hoverinfo="skip",
        ))

fig.update_layout(
    height=480, plot_bgcolor="white",
    xaxis_title="data", yaxis_title="anomalia Niño 3.4 (degC)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    font=dict(family="Inter, -apple-system, Segoe UI, Arial, sans-serif"),
    margin=dict(l=10, r=10, t=10, b=10),
)
fig.update_yaxes(range=[-3.0, 3.0], gridcolor="#eeeeee")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.divider()

st.subheader("3. Construcao do intervalo de confianca")
st.markdown(
    """
O **IC 90%** representado pela faixa em torno da media e empirico, derivado
do espalhamento entre as 10 sementes do ensemble. Procedimento:

1. Para cada seed, treinar o modelo vencedor com a serie completa de treino.
2. Predizer o valor de SST anomaly em t+h.
3. Tirar quantis 5%, 25%, 50%, 75% e 95% sobre as 10 previsoes.

A largura do IC tipicamente cresce com o horizonte de previsao h, refletindo
o aumento de incerteza intrinseca do sistema. Em ENSO, o erro RMSE empirico
e da ordem de 0,3 degC em h=1 e pode ultrapassar 1,0 degC em h=6 nas
configuracoes testadas.
"""
)

with st.expander("Como atualizar a previsao oficial CPC/IRI"):
    st.markdown(
        """
O CPC e o IRI publicam mensalmente a previsao consensual do ENSO:

- [CPC ENSO Advisory](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/)
- [IRI ENSO Quick Look](https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/)

Para atualizar aqui, edite `data/raw/official_forecast.csv` com as colunas:
`target_month, season, mean, q05, q95, el_nino_prob, neutral_prob, la_nina_prob, source, issued`.
        """
    )
