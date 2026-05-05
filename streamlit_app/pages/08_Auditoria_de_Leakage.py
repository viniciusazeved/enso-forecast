"""Pagina educacional: auditoria de leakage e por que o ONI nao deve ser alvo direto."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from shared.components import aplicar_estilo, header_pagina, kpi_card
from utils import load_master

st.set_page_config(page_title="Auditoria de leakage - ENSO", page_icon="🌊", layout="wide")
aplicar_estilo()

header_pagina(
    "Auditoria de leakage",
    "Por que usar ONI como alvo (e como feature defasada) infla artificialmente "
    "as metricas de qualquer modelo de previsao.",
)

# 1. O problema
st.subheader("1. O problema do alvo errado")
st.markdown(
    """
O **Indice Oceanico de El Nino (ONI)** e, por definicao:

> ONI(t) = media (SST_anom Niño 3.4 em t-1, t, t+1)

E uma **media movel centrada** de 3 meses. Por isso, o ONI no mes anterior
compartilha **2 dos 3 componentes** com o ONI atual:

- ONI(t-1) = media(SST(t-2), SST(t-1), SST(t))
- ONI(t)   = media(SST(t-1), SST(t),   SST(t+1))

Quando voce treina um modelo para prever **ONI(t) usando ONI(t-1)**, parte da
resposta ja esta na pergunta. Resultado: **R² inflado** que nao reflete
capacidade preditiva real, so estrutura matematica do alvo.
"""
)

st.divider()

# 2. Demonstracao com dados reais
st.subheader("2. Demonstracao com dados reais")

df = load_master(since="1982-01-01")
oni  = df["oni"].dropna() if "oni" in df.columns else None
nino = df["nino34_anom"].dropna() if "nino34_anom" in df.columns else None

if oni is None or nino is None:
    st.warning("ONI ou nino34_anom nao disponiveis no dataset mestre.")
    st.stop()

common = oni.index.intersection(nino.index)
oni  = oni.loc[common]
nino = nino.loc[common]

o1 = oni.shift(1).dropna()
o0 = oni.loc[o1.index]
n1 = nino.shift(1).dropna()
n0 = nino.loc[n1.index]
c_oni = float(np.corrcoef(o1, o0)[0, 1])
c_nino = float(np.corrcoef(n1, n0)[0, 1])

c1, c2, c3 = st.columns(3)
with c1:
    kpi_card("corr(ONI(t), ONI(t-1))", f"{c_oni:.4f}",
             help="Inflada pela media movel")
with c2:
    kpi_card("corr(nino34_anom(t), nino34_anom(t-1))", f"{c_nino:.4f}",
             help="Limite real (so persistencia natural)")
with c3:
    kpi_card("Gap (leakage estrutural)", f"{c_oni - c_nino:+.4f}",
             tone="warn",
             help="Quanto da 'previsao' do ONI vem da media movel, nao do modelo")

st.markdown(
    f"O ONI tem auto-correlacao **{(c_oni-c_nino)*100:+.2f} pontos percentuais** "
    f"acima da serie subjacente. Esse 'extra' e estrutural - nao e skill do modelo, "
    f"e um artefato da media movel. Qualquer modelo treinado para prever ONI a "
    f"partir de ONI defasado herda essa inflacao.\n\n"
    f"Para evitar essa inflacao, **usamos `nino34_anom` como alvo** (sem media movel) "
    f"e excluimos qualquer feature que contenha o alvo via media movel centrada."
)

st.divider()

# 3. Visualizacao
st.subheader("3. Visualizacao do overlap estrutural")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=nino.tail(36).index, y=nino.tail(36).values, mode="lines+markers",
    name="nino34_anom (mensal)", line=dict(color="#1c2a3a", width=2),
))
fig.add_trace(go.Scatter(
    x=oni.tail(36).index, y=oni.tail(36).values, mode="lines+markers",
    name="ONI (media movel 3m)", line=dict(color="#1f4e79", width=2),
))
fig.update_layout(
    height=380, plot_bgcolor="white",
    yaxis_title="anomalia (degC)", xaxis_title="data",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    font=dict(family="Inter, -apple-system, Segoe UI, Arial, sans-serif"),
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.caption(
    "O ONI segue o nino34_anom mas e mais suave - exatamente porque cada ponto "
    "do ONI carrega informacao dos vizinhos. Essa suavizacao e desejavel pra "
    "diagnostico oficial (define eventos), mas problematica pra usar como alvo "
    "de modelo de previsao."
)

st.divider()

# 4. Outras checagens
st.subheader("4. Outras checagens automaticas no pipeline")
st.markdown(
    """
Cada run de training executa as seguintes assertions antes de treinar:

1. **Splits monotonicos**. Treino sempre antecede validacao, validacao antecede teste.
   Sem indices em comum entre conjuntos.
2. **Scaler ajustado SO no treino do fold**. Mesmo MinMaxScaler/StandardScaler
   e refitado para cada fold; a media/std do teste nunca toca o treino.
3. **Alvo nao esta entre features**. Trivial mas indispensavel.
4. **Nenhuma feature tem corr aproximadamente 1 com o alvo no instante t**.
   Heuristica de leakage estrutural (janela centrada, shift errado).
5. **Toda feature `<var>_lag<k>` tem k maior ou igual a 1**. Sem leitura do futuro.

Falha em qualquer um para o pipeline (`LeakageError`).
O codigo das checagens esta em `src/enso/data/leakage.py`.
"""
)

st.divider()

# 5. Implicacao quantitativa
st.subheader("5. Implicacao quantitativa para metricas de previsao")
st.markdown(
    """
A escolha de alvo e o regime de validacao definem a faixa de R² esperada para
um modelo de previsao do ENSO ser informativo. Considerando o limite teorico
imposto pela autocorrelacao natural da serie:

| Configuracao | R² esperado para baseline | Limiar para skill genuino |
|---|---|---|
| Alvo = ONI(t), feature = ONI(t-1), horizonte 1 | ~0,95 (artefato MA) | requer R² > 0,97 |
| Alvo = nino34_anom(t), feature = nino34_anom(t-1), h=1 | ~0,89 (persistencia) | R² > 0,90 = ganho marginal |
| Alvo = nino34_anom(t+6), walk-forward CV | ~0,15 (clima) | R² > 0,40 = expressivo |

A persistencia simples (`y_hat(t+h) = y(t)`) e o baseline obrigatorio. Modelos
que nao a superam por margem estatisticamente significativa nao agregam
informacao - apenas reproduzem a inercia natural do sistema.
"""
)
