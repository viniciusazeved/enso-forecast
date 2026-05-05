"""Contexto historico - estatisticas descritivas dos eventos ENSO passados."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shared.components import aplicar_estilo, header_pagina, kpi_card
from shared.style import fase_para_value
from utils import load_master

st.set_page_config(page_title="Contexto historico - ENSO", page_icon="🌊", layout="wide")
aplicar_estilo()

header_pagina(
    "Contexto historico",
    "Estatisticas descritivas dos eventos passados (1982-presente). Permite "
    "calibrar a leitura do estado atual e do forecast em relacao ao registro historico.",
)

df = load_master()
s = df["nino34_anom"].dropna()

# 1. Top eventos
st.subheader("1. Top eventos por intensidade")
st.markdown(
    "Os 10 meses com maior anomalia positiva (eventos El Nino) e os 10 meses "
    "com maior anomalia negativa (eventos La Nina) na regiao Niño 3.4 desde 1982. "
    "A categoria oficial considera intensidade do pico durante o evento, nao "
    "valores mensais isolados."
)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Top-10 anomalias positivas**")
    top_pos = s.nlargest(10).reset_index()
    top_pos.columns = ["mes", "anomalia (degC)"]
    top_pos["categoria"] = top_pos["anomalia (degC)"].apply(lambda v: fase_para_value(v)[0])
    top_pos["mes"] = top_pos["mes"].dt.strftime("%b/%Y")
    st.dataframe(top_pos, use_container_width=True, hide_index=True)
with c2:
    st.markdown("**Top-10 anomalias negativas**")
    top_neg = s.nsmallest(10).reset_index()
    top_neg.columns = ["mes", "anomalia (degC)"]
    top_neg["categoria"] = top_neg["anomalia (degC)"].apply(lambda v: fase_para_value(v)[0])
    top_neg["mes"] = top_neg["mes"].dt.strftime("%b/%Y")
    st.dataframe(top_neg, use_container_width=True, hide_index=True)

st.markdown(
    "Os tres eventos com pico acima de +2,5 degC (El Nino muito forte) "
    "ocorreram em **1982-83, 1997-98 e 2015-16**. La Ninas com pico abaixo "
    "de -2,0 degC sao raras na serie."
)

st.divider()

# 2. Estado atual em contexto
st.subheader("2. Posicionamento estatistico do estado atual")

last_val = float(s.iloc[-1])
last_date = s.index[-1]
pct_above = float((s.values > last_val).mean() * 100)
pico_historico_pos = float(s.max())
pico_historico_neg = float(s.min())

c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Leitura atual", f"{last_val:+.2f} degC", help=f"em {last_date:%b/%Y}")
with c2:
    kpi_card("Percentil historico", f"{100 - pct_above:.0f}",
             suffix=f"% (so {pct_above:.0f}% dos meses foram mais quentes)")
with c3:
    kpi_card("Maximo historico", f"{pico_historico_pos:+.2f} degC",
             help="Nov/2015")
with c4:
    kpi_card("Minimo historico", f"{pico_historico_neg:+.2f} degC",
             help="Jan/2000")

st.divider()

# 3. Distribuicao
st.subheader("3. Distribuicao das anomalias mensais")
fig = go.Figure()
fig.add_trace(go.Histogram(
    x=s.values, nbinsx=50, marker_color="#7faedd",
    opacity=0.85, name="meses observados",
))
fig.add_vline(
    x=last_val, line=dict(color="#cb181d", width=3, dash="dash"),
    annotation_text=f"atual ({last_val:+.2f})",
    annotation_position="top right",
)
fig.add_vline(
    x=2.0, line=dict(color="#67000d", width=1.2, dash="dot"),
    annotation_text="+2.0 degC", annotation_position="top right",
)
fig.add_vline(
    x=-2.0, line=dict(color="#08306b", width=1.2, dash="dot"),
    annotation_text="-2.0 degC", annotation_position="top left",
)
fig.update_layout(
    height=360, plot_bgcolor="white",
    xaxis_title="anomalia Niño 3.4 (degC)", yaxis_title="contagem de meses",
    font=dict(family="Inter, -apple-system, Segoe UI, Arial, sans-serif"),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.caption(
    f"Distribuicao das anomalias mensais da Niño 3.4 desde 1982 (n={len(s)} meses). "
    "Linha vermelha tracejada: leitura atual. Linhas pontilhadas: limites das "
    "categorias 'muito forte' (+/- 2,0 degC)."
)

st.divider()

# 4. Frequencia de fases
st.subheader("4. Frequencia historica das fases")

cont = pd.Series([fase_para_value(v)[0] for v in s.values]).value_counts()
total = len(s)
pct = (cont / total * 100).round(1)
df_cont = pd.DataFrame({
    "Categoria": cont.index,
    "Meses": cont.values,
    "% do registro": pct.values,
}).sort_values("Meses", ascending=False)
st.dataframe(df_cont, use_container_width=True, hide_index=True)

st.markdown(
    "A categoria **Neutro** (anomalia entre -0,5 e +0,5) cobre a maior parte "
    "do registro historico, como esperado para uma oscilacao em torno do zero."
)

st.divider()

# 5. Decadas
st.subheader("5. Estatisticas por decada")
df_dec = pd.DataFrame({"anom": s.values, "decada": (s.index.year // 10) * 10})
agg = df_dec.groupby("decada").agg(
    media=("anom", "mean"),
    mediana=("anom", "median"),
    desvio=("anom", "std"),
    minimo=("anom", "min"),
    maximo=("anom", "max"),
    n_meses=("anom", "count"),
).round(2)
st.dataframe(agg, use_container_width=True)

st.caption(
    "Nenhum teste de tendencia foi aplicado - sao estatisticas descritivas "
    "agregadas por decada, sujeitas a variancia amostral em decadas com "
    "menos de 120 meses."
)
