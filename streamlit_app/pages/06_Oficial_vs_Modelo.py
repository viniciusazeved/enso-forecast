"""Comparacao explicita: previsao oficial CPC/IRI vs nosso modelo."""
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

st.set_page_config(page_title="Oficial vs modelo - ENSO", page_icon="🌊", layout="wide")
aplicar_estilo()

header_pagina(
    "Oficial CPC/IRI vs nosso modelo",
    "A previsao consensual do IRI (publicada no plume mensal do CPC) e a "
    "referencia internacional. Aqui ela aparece lado a lado com nosso ensemble.",
)

df = load_master()
fc_ours = load_forecast()
fc_off  = load_official_forecast()

if fc_off.empty:
    st.warning("Sem dados oficiais carregados. Edite `data/raw/official_forecast.csv`.")
    st.stop()

issued = fc_off["issued"].dropna().max()
if pd.notna(issued):
    st.markdown(f"**Boletim oficial usado**: publicado em {issued:%d/%b/%Y} (IRI Quick Look).")

# 1. Tabela oficial
st.subheader("1. Previsao oficial - tabela")
show = fc_off.copy()
show["target_month"] = show["target_month"].dt.strftime("%b/%Y")
show["fase prevista"] = fc_off["mean"].apply(lambda v: fase_para_value(v)[0])
st.dataframe(
    show[["target_month", "season", "mean", "q05", "q95",
          "el_nino_prob", "neutral_prob", "la_nina_prob", "fase prevista", "source"]]
    .rename(columns={
        "target_month": "alvo",
        "mean":  "media (degC)",
        "el_nino_prob": "P(El Nino) %",
        "neutral_prob": "P(Neutro) %",
        "la_nina_prob": "P(La Nina) %",
    }),
    use_container_width=True, hide_index=True,
)

st.divider()

# 2. Curva
st.subheader("2. Curva: oficial vs nosso modelo")
serie = df["nino34_anom"].dropna().tail(36)

fig = go.Figure()
for nome, lo, hi, cor in FAIXAS_FASE:
    lo_v = max(lo, -3.0); hi_v = min(hi, 3.0)
    if hi_v <= lo_v: continue
    fig.add_hrect(y0=lo_v, y1=hi_v, fillcolor=cor, opacity=0.15, line_width=0)

fig.add_trace(go.Scatter(
    x=serie.index, y=serie.values, mode="lines",
    name="observado", line=dict(color="#1c2a3a", width=2.2),
))

off_sorted = fc_off.sort_values("target_month")
fig.add_trace(go.Scatter(
    x=off_sorted["target_month"], y=off_sorted["mean"],
    mode="lines+markers", name="IRI/CPC (oficial)",
    line=dict(color="#d62728", width=2.5, dash="dot"),
    marker=dict(size=10, symbol="x"),
))
if {"q05", "q95"}.issubset(off_sorted.columns):
    fig.add_trace(go.Scatter(
        x=list(off_sorted["target_month"]) + list(off_sorted["target_month"][::-1]),
        y=list(off_sorted["q95"]) + list(off_sorted["q05"][::-1]),
        fill="toself", fillcolor="rgba(214,39,40,0.18)",
        line=dict(color="rgba(0,0,0,0)"), name="IC oficial 90%", hoverinfo="skip",
    ))

if not fc_ours.empty:
    last_date = serie.index[-1]
    last_val  = float(serie.iloc[-1])
    ours = fc_ours.sort_values("horizon")
    xs = [last_date] + ours["target"].tolist()
    ys = [last_val] + ours["mean"].tolist()
    qlo = [last_val] + ours["q05"].tolist()
    qhi = [last_val] + ours["q95"].tolist()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers", name="nosso ensemble",
        line=dict(color="#1f4e79", width=2.5, dash="dash"),
        marker=dict(size=9, symbol="diamond"),
    ))
    fig.add_trace(go.Scatter(
        x=xs + xs[::-1], y=qhi + qlo[::-1],
        fill="toself", fillcolor="rgba(31,78,121,0.20)",
        line=dict(color="rgba(0,0,0,0)"), name="IC nosso 90%", hoverinfo="skip",
    ))

fig.update_layout(
    height=480, plot_bgcolor="white",
    xaxis_title="data", yaxis_title="anomalia Niño 3.4 (degC)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    font=dict(family="Inter, -apple-system, Segoe UI, Arial, sans-serif"),
    margin=dict(l=10, r=10, t=10, b=10),
)
fig.update_yaxes(range=[-2.5, 3.0], gridcolor="#eeeeee")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.divider()

# 3. Probabilidades
st.subheader("3. Probabilidades de fase (oficial)")
prob_cols = ["el_nino_prob", "neutral_prob", "la_nina_prob"]
if all(c in fc_off.columns for c in prob_cols):
    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(x=fc_off["season"], y=fc_off["el_nino_prob"],
                           name="El Nino", marker_color="#cb181d"))
    fig_p.add_trace(go.Bar(x=fc_off["season"], y=fc_off["neutral_prob"],
                           name="Neutro", marker_color="#bdbdbd"))
    fig_p.add_trace(go.Bar(x=fc_off["season"], y=fc_off["la_nina_prob"],
                           name="La Nina", marker_color="#2171b5"))
    fig_p.update_layout(
        barmode="stack", height=360,
        plot_bgcolor="white", yaxis_title="probabilidade (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="Inter, -apple-system, Segoe UI, Arial, sans-serif"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_p, use_container_width=True, config={"displayModeBar": False})

st.divider()

# 4. Sintese quantitativa
st.subheader("4. Sintese quantitativa")
if not fc_off.empty:
    pico_off = float(fc_off["mean"].max())
    pico_off_season = fc_off.loc[fc_off["mean"].idxmax(), "season"]
    iCsup_off = float(fc_off.loc[fc_off["mean"].idxmax(), "q95"])
    st.markdown(
        f"""
- Pico previsto pela CPC/IRI: **{pico_off:+.2f} degC em {pico_off_season}**.
- IC 90% superior no pico: **{iCsup_off:+.2f} degC**.
- Distancia ao limiar de El Nino muito forte (+2,0 degC): **{2.0 - pico_off:+.2f} degC**
  na media; **{2.0 - iCsup_off:+.2f} degC** no IC superior.
- Probabilidade media de El Nino no horizonte coberto:
  **{int(fc_off['el_nino_prob'].mean())}%** (variando de
  {int(fc_off['el_nino_prob'].min())}% a {int(fc_off['el_nino_prob'].max())}%).

Variaveis de monitoramento secundario citadas pelo CPC para acompanhar a
evolucao do cenario nas proximas iteracoes do plume:

- **Subsuperficie do Pacifico equatorial** - conteudo de calor abaixo da
  superficie disponivel para emergir.
- **Onda Kelvin oceanica** - propagacao de anomalias de leste para oeste.
- **Vento zonal de superficie** (alisios) na faixa equatorial central.
"""
    )
