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
    "O modelo vencedor por horizonte foi eleito por score unificado (ver pagina "
    "**Modelos**). A coluna **R² CV** indica o R² medio do vencedor na "
    "validacao cruzada walk-forward; valores **negativos sinalizam ausencia "
    "de skill** (a previsao nao agrega informacao alem da media historica)."
)

# R² da CV por horizonte (computado em runs/train_full_v1_completo)
R2_CV = {1: 0.62, 2: 0.35, 3: -0.07, 4: -0.26, 5: -0.19, 6: -0.15}

show = fc.copy()
show["target"] = show["target"].dt.strftime("%b/%Y")
show["fase prevista"] = fc["mean"].apply(lambda v: fase_para_value(v)[0])
show["r2_cv"] = fc["horizon"].map(R2_CV)
show["skill"] = show["r2_cv"].apply(
    lambda r: "validado" if r >= 0.30 else ("borderline" if r >= 0 else "sem skill")
)

st.dataframe(
    show[["horizon", "target", "model", "mean", "q05", "q25", "q75", "q95",
          "n_seeds", "r2_cv", "skill", "fase prevista"]].rename(columns={
        "horizon": "h (meses)",
        "target":  "alvo",
        "model":   "modelo",
        "mean":    "media",
        "n_seeds": "seeds",
        "r2_cv":   "R² CV",
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

# Separa os horizontes com skill (h<=2) dos sem skill (h>=3)
fc_sk = fc[fc["horizon"] <= 2].sort_values("horizon")
fc_no = fc[fc["horizon"] >= 3].sort_values("horizon")

# Trecho com skill: solido + IC
ref_sk = pd.DataFrame({
    "date": [last_date] + fc_sk["target"].tolist(),
    "mean": [last_val] + fc_sk["mean"].tolist(),
    "q05":  [last_val] + fc_sk["q05"].tolist(),
    "q95":  [last_val] + fc_sk["q95"].tolist(),
})
fig.add_trace(go.Scatter(
    x=ref_sk["date"], y=ref_sk["mean"], mode="lines+markers",
    name="nosso ensemble (h<=2, skill validado)",
    line=dict(color="#1f4e79", width=2.5, dash="dash"),
    marker=dict(size=10, symbol="diamond"),
))
fig.add_trace(go.Scatter(
    x=list(ref_sk["date"]) + list(ref_sk["date"][::-1]),
    y=list(ref_sk["q95"])  + list(ref_sk["q05"][::-1]),
    fill="toself", fillcolor="rgba(31,78,121,0.20)",
    line=dict(color="rgba(0,0,0,0)"), name="IC nosso 90%", hoverinfo="skip",
))

# Trecho sem skill: tracejado claro, sem IC, com aviso visual
if not fc_no.empty:
    last_sk = fc_sk.iloc[-1] if not fc_sk.empty else None
    x0_no = last_sk["target"] if last_sk is not None else last_date
    y0_no = float(last_sk["mean"]) if last_sk is not None else last_val
    ref_no = pd.DataFrame({
        "date": [x0_no] + fc_no["target"].tolist(),
        "mean": [y0_no] + fc_no["mean"].tolist(),
    })
    fig.add_trace(go.Scatter(
        x=ref_no["date"], y=ref_no["mean"], mode="lines+markers",
        name="nosso ensemble (h>=3, sem skill validado)",
        line=dict(color="#9eb3c9", width=1.5, dash="dot"),
        marker=dict(size=8, symbol="diamond-open"),
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

st.subheader("3. Skill por horizonte e mean reversion")
st.markdown(
    """
Os modelos sao avaliados em walk-forward CV antes do retreino final. Para o
run atual (`full_v1_completo`):

- **h=1 e h=2**: R² na CV positivo (0.62 e 0.35 respectivamente). O ensemble
  agrega informacao alem da media historica - **skill validado**.
- **h>=3**: R² na CV negativo. O ensemble e estatisticamente pior que prever
  a climatologia mensal nesses horizontes. Os valores reportados sao mantidos
  na tabela por completude, mas marcados como **sem skill**.

Quando R² da CV e negativo, redes neurais tendem a entrar em **mean reversion
patologico**: as previsoes encolhem em direcao a media da serie de treino
(proxima de 0 degC para `nino34_anom`). O h=6 do run atual ilustra isso: a
media prevista (~+0.18 degC) esta proxima de zero, enquanto a CPC/IRI projeta
+1.57 degC - a divergencia nao reflete uma discordancia analitica, mas a
ausencia de skill do ensemble nesse horizonte.

Caminhos para estender o horizonte util:

- Adicionar features subsuperficiais (HCA, WWV) como em Ham et al. 2019.
- Substituir o vencedor sem skill por persistencia ou climatologia mensal
  como fallback para h>=3.
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
