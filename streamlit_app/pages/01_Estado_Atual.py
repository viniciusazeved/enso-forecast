"""Estado atual do ENSO em 2026 - snapshot didatico das 4 regioes Niño."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shared.components import aplicar_estilo, fase_badge, header_pagina, kpi_card
from shared.style import FAIXAS_FASE, fase_para_value
from utils import load_master

st.set_page_config(page_title="Estado atual - ENSO", page_icon="🌊", layout="wide")
aplicar_estilo()

header_pagina(
    "Estado atual do ENSO",
    "Onde estamos agora, qual a velocidade de mudanca e como isso se compara "
    "com transicoes historicas para El Nino.",
)

df = load_master()

# 1. Snapshot por regiao
st.subheader("1. Snapshot por regiao")
st.markdown(
    """
O Pacifico equatorial e monitorado em 4 caixas oceanicas (regioes Niño 1+2,
3, 3.4 e 4). Olhando as 4 simultaneamente da pra entender o **tipo** de El
Nino que esta se formando, nao so se ha um.
"""
)

last_date = df["nino34_anom"].dropna().index[-1]
st.caption(f"Leituras mais recentes: {last_date:%b/%Y}")

c1, c2, c3, c4 = st.columns(4)
configs = [
    (c1, "Niño 1+2",  "nino12_anom",  "Costa do Peru/Equador. Primeiro lugar a esquentar em El Nino canonico."),
    (c2, "Niño 3",    "nino3_anom",   "Pacifico tropical leste. El Nino classico aparece aqui depois do Niño 1+2."),
    (c3, "Niño 3.4",  "nino34_anom",  "Centro do Pacifico. Referencia oficial - define o ONI."),
    (c4, "Niño 4",    "nino4_anom",   "Pacifico central-oeste. Captura padrao Modoki."),
]
for col, label, key, help_text in configs:
    if key in df.columns:
        s = df[key].dropna()
        if len(s):
            v = float(s.iloc[-1])
            tone = "warn" if v > 1.0 else ("good" if v < -0.5 else "neutral")
            with col:
                kpi_card(label, f"{v:+.2f} degC", help=help_text, tone=tone)

st.markdown(
    """
**Como interpretar o padrao por regiao:**

- **Aquecimento progressivo de leste para oeste** (Niño 1+2 > Niño 3 > Niño 3.4):
  El Nino **canonico**, com origem em **onda Kelvin** oceanica. E o tipo classico,
  como 1997-98 e 2015-16.
- **Aquecimento concentrado no centro** (Niño 4 > Niño 3.4): El Nino **Modoki**,
  com impactos diferentes na America do Sul (mais seca no Sudeste, menos chuva
  no Sul).
- **Anomalias proximas de zero em todas**: fase neutra ou transicao.
"""
)

st.divider()

# 2. Velocidade da transicao
st.subheader("2. Velocidade da transicao recente")
st.markdown(
    "O quanto a Niño 3.4 mudou nos ultimos meses ajuda a entender se a transicao "
    "para El Nino esta acelerando ou estabilizando."
)

n = st.slider("Janela (meses)", 3, 36, 12)
sub = df["nino34_anom"].dropna().tail(n)
deltas = sub.diff().dropna()
total = float(sub.iloc[-1] - sub.iloc[0])

c1, c2, c3 = st.columns(3)
with c1:
    kpi_card(f"Variacao em {n} meses", f"{total:+.2f} degC")
with c2:
    kpi_card("Maior salto mensal", f"{deltas.max():+.2f} degC",
             help=f"em {deltas.idxmax():%b/%Y}")
with c3:
    kpi_card("Maior queda mensal", f"{deltas.min():+.2f} degC",
             help=f"em {deltas.idxmin():%b/%Y}")

fig = go.Figure()
for nome, lo, hi, cor in FAIXAS_FASE:
    lo_v = max(lo, -3.0); hi_v = min(hi, 3.0)
    if hi_v <= lo_v: continue
    fig.add_hrect(y0=lo_v, y1=hi_v, fillcolor=cor, opacity=0.15, line_width=0)
fig.add_trace(go.Scatter(
    x=sub.index, y=sub.values, mode="lines+markers",
    name="Niño 3.4 (mensal)", line=dict(color="#1c2a3a", width=2.2),
    marker=dict(size=7),
))
fig.update_layout(
    height=380, plot_bgcolor="white",
    xaxis_title="data", yaxis_title="anomalia (degC)",
    font=dict(family="Inter, -apple-system, Segoe UI, Arial, sans-serif"),
    margin=dict(l=10, r=10, t=10, b=10),
)
fig.update_yaxes(range=[-3.0, 3.0], gridcolor="#eeeeee")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.divider()

# 3. Comparacao com transicoes historicas
st.subheader("3. Comparacao com transicoes historicas")
st.markdown(
    "Quando a Niño 3.4 cruza o limiar de El Nino (+0,5 degC) **subindo**, o que "
    "costuma acontecer nos meses seguintes? Cada linha cinza abaixo e uma "
    "transicao passada, alinhada no mesmo ponto zero. A linha vermelha mostra "
    "a trajetoria recente. Permite ver se o ritmo atual se parece com algum "
    "episodio canonico."
)

s = df["nino34_anom"].dropna()
crossings = []
for i in range(1, len(s)):
    if s.iloc[i - 1] < 0.5 and s.iloc[i] >= 0.5:
        crossings.append(s.index[i])

window = pd.DataFrame()
for cr in crossings:
    start = cr - pd.DateOffset(months=6)
    end   = cr + pd.DateOffset(months=18)
    seg = s.loc[(s.index >= start) & (s.index <= end)]
    if len(seg) < 12:
        continue
    rel_months = ((seg.index.year - cr.year) * 12 + (seg.index.month - cr.month)).astype(int)
    window[cr.strftime("%b/%Y")] = pd.Series(seg.values, index=rel_months)

window["agora (recente)"] = pd.Series(
    s.tail(12).values, index=range(-len(s.tail(12)) + 1, 1),
)

fig_overlay = go.Figure()
for col in window.columns:
    is_now = col == "agora (recente)"
    fig_overlay.add_trace(go.Scatter(
        x=window.index, y=window[col], mode="lines",
        name=col,
        line=dict(width=3 if is_now else 1.0, color="#cb181d" if is_now else None),
        opacity=1.0 if is_now else 0.45,
    ))
fig_overlay.add_hline(
    y=0.5, line=dict(color="#fb6a4a", dash="dash"),
    annotation_text="limiar El Nino (+0.5)", annotation_position="top right",
)
fig_overlay.add_hline(
    y=2.0, line=dict(color="#67000d", dash="dot"),
    annotation_text="super El Nino (+2.0)", annotation_position="top right",
)
fig_overlay.update_layout(
    height=440, plot_bgcolor="white",
    xaxis_title="meses relativos a cruzar +0,5 degC (negativo = antes)",
    yaxis_title="anomalia Niño 3.4 (degC)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                font=dict(size=10)),
    font=dict(family="Inter, -apple-system, Segoe UI, Arial, sans-serif"),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig_overlay, use_container_width=True, config={"displayModeBar": False})

st.caption(
    "Cada linha cinza e uma vez que a Niño 3.4 cruzou +0,5 degC subindo "
    "(de 1982 a 2024). A linha vermelha grossa mostra a trajetoria atual. "
    "Nao e previsao - e referencia historica para calibrar expectativa."
)

st.divider()

st.subheader("4. Sintese")
st.markdown(
    """
As tres analises desta pagina sao descritivas:

- **Snapshot por regiao** caracteriza a estrutura espacial do aquecimento
  (canonico vs Modoki).
- **Velocidade** quantifica a magnitude e direcao da mudanca recente.
- **Analogos historicos** fornece referencia empirica para episodios passados
  com partida similar.

Nenhuma delas e previsao formal. Para o forecast quantitativo com IC, ver
pagina **Forecast**.
"""
)
