"""ENSO Forecast - relato narrativo (entry point Streamlit).

Para rodar localmente:
    cd D:/Projetos/ENSO
    uv run streamlit run streamlit_app/Inicio.py
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shared.components import (
    aplicar_estilo,
    fase_badge,
    header_pagina,
    kpi_card,
)
from shared.style import FAIXAS_FASE, fase_para_value
from utils import load_forecast, load_master

sys.path.insert(0, str(APP_DIR.parent / "src"))
from enso.forecast.official import load_official_forecast  # noqa: E402

st.set_page_config(
    page_title="Panorama ENSO 2026",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)
aplicar_estilo()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🌊 Panorama ENSO")
    st.caption("**E**l **N**ino-**S**outhern **O**scillation")
    st.divider()
    st.markdown(
        """
**Vinicius Azevedo**
Doutorando em Recursos Hidricos, Energeticos e Ambientais
FECFAU - UNICAMP
Orientador: Prof. Hugo de Oliveira Fagundes
"""
    )
    st.divider()
    st.markdown(
        "<small>Dashboard companheiro ao trabalho publicado em "
        "<i>Scientia Plena</i> 22 (2026), com auditoria de leakage, ensemble "
        "multi-seed e comparacao com a previsao consensual CPC/IRI.</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Cabecalho
# ---------------------------------------------------------------------------
st.markdown(
    """
# Panorama ENSO - 2026

<p style='color:#5a6a7a; font-size:15px; margin-top:0; margin-bottom:18px;
          line-height:1.5;'>
Sintese mensal do estado do El Nino-Oscilacao Sul, com previsao consensual
CPC/IRI e saida de uma bateria de redes neurais comparadas em walk-forward CV.
</p>
""",
    unsafe_allow_html=True,
)

# Carrega dados
df = load_master()
fc_ours = load_forecast()
fc_off = load_official_forecast()

last_date = df["nino34_anom"].dropna().index[-1]
last_val = float(df["nino34_anom"].dropna().iloc[-1])
fase_atual, _ = fase_para_value(last_val)

# ---------------------------------------------------------------------------
# 1. Definicao do fenomeno
# ---------------------------------------------------------------------------
st.subheader("1. Definicao do fenomeno")
st.markdown(
    """
**ENSO** (*El Nino - Southern Oscillation*) e um modo acoplado oceano-atmosfera
do Pacifico equatorial caracterizado por tres estados: **El Nino** (anomalia
positiva de SST na faixa central-leste, enfraquecimento dos alisios),
**La Nina** (anomalia negativa, alisios reforcados) e **Neutro** (sem desvio
estatisticamente significativo).

A oscilacao tem periodicidade irregular (2 a 7 anos) e amplitude variavel.
Nao e um fenomeno induzido pelo aquecimento global, mas sua frequencia e
intensidade podem ser moduladas por ele (Cai et al., 2015).

A definicao operacional adotada pela NOAA usa o **ONI** (*Oceanic Niño Index*),
media movel trimestral da anomalia mensal de SST na regiao Niño 3.4
(170 W a 120 W, +/-5 graus de latitude). Os limiares sao:

| ONI | Categoria |
|---|---|
| ≥ +0,5 degC por 5 trimestres consecutivos | El Nino |
| ≤ -0,5 degC por 5 trimestres consecutivos | La Nina |
| dentro de [-0,5; +0,5] | Neutro |

Categorias de intensidade (pico do ONI durante o evento):

| Faixa | Categoria |
|---|---|
| +0,5 a +1,0 | El Nino fraco |
| +1,0 a +1,5 | El Nino moderado |
| +1,5 a +2,0 | El Nino forte |
| acima de +2,0 | El Nino muito forte |
"""
)

# ---------------------------------------------------------------------------
# 2. Implicacoes para o territorio brasileiro
# ---------------------------------------------------------------------------
st.subheader("2. Implicacoes para o territorio brasileiro")
st.markdown(
    """
Por meio de teleconexoes atmosfericas, o ENSO modula a probabilidade de
anomalias de precipitacao e temperatura em diversas regioes do mundo
(Trenberth et al., 1998). No Brasil, os padroes tipicos reportados na
literatura sao:

| Regiao | Em El Nino | Em La Nina |
|---|---|---|
| Sul (RS, SC, PR) | precipitacao acima da media | tendencia de seca |
| Sudeste/Centro-Oeste | seca/calor anomalo no inverno-primavera | tendencia de chuva normal a acima |
| Nordeste (sertao) | seca severa | precipitacao acima da media |
| Norte (Amazonia leste) | seca, risco de queimadas | precipitacao normal a acima |

A magnitude do impacto regional depende da fase do ENSO (canonica vs Modoki,
ver pagina **Estado atual**), da interacao com outros modos climaticos
(IOD, AMO, MJO) e de processos locais. A previsao do estado do ENSO com
horizonte de 3 a 6 meses tem aplicacao operacional em planejamento de
safra, despacho hidreletrico e gestao de recursos hidricos.
"""
)

# ---------------------------------------------------------------------------
# 3. Variaveis monitoradas
# ---------------------------------------------------------------------------
st.subheader("3. Variaveis monitoradas")
st.markdown(
    """
A variavel-alvo deste dashboard e a **anomalia mensal de SST na regiao Niño 3.4**
(`nino34_anom`), nao o ONI. A escolha evita o overlap estrutural da media
movel trimestral usada no ONI - documentado na pagina **Auditoria de leakage**.

As features preditoras incluem indices oceanicos e atmosfericos publicos da
NOAA, CPC e SILSO (1950-presente, frequencia mensal):

- **SST e anomalias** das regioes Niño 1+2, 3, 3.4 e 4 (CPC sstoi).
- **SOI** (Southern Oscillation Index, NOAA PSL).
- **MEI v2** (Multivariate ENSO Index, NOAA PSL).
- **OLR** (Outgoing Longwave Radiation, NOAA PSL).
- **PNA** (Pacific North American pattern, NOAA PSL).
- **PDO** (Pacific Decadal Oscillation, NCEI ERSSTv5).
- **IOD/DMI** (Indian Ocean Dipole, NOAA PSL).
- **QBO** (Quasi-Biennial Oscillation, NOAA PSL).
- **TNI** (Trans-Niño Index, NOAA PSL).
- **Sunspots** (numero mensal SILSO Belgica) como proxy solar.

Definicoes detalhadas estao na pagina **Glossario**.
"""
)

# ---------------------------------------------------------------------------
# 4. Estado atual
# ---------------------------------------------------------------------------
st.subheader("4. Estado atual")

c_a, c_b, c_c = st.columns([1.2, 1, 1])
with c_a:
    st.markdown(f"**Ultima leitura disponivel: {last_date:%b/%Y}**")
    fase_badge(last_val)
    st.caption(
        f"Anomalia mensal de SST na Niño 3.4 = {last_val:+.2f} degC. "
        f"Limiar oficial de evento El Nino: +0,5 degC."
    )

with c_b:
    if "oni" in df.columns:
        oni = df["oni"].dropna()
        oni_val = float(oni.iloc[-1])
        oni_date = oni.index[-1]
        kpi_card(
            "ONI (media trimestral)",
            f"{oni_val:+.2f} degC",
            help=f"Mes central: {oni_date:%b/%Y}",
        )

with c_c:
    nino12 = float(df["nino12_anom"].dropna().iloc[-1]) if "nino12_anom" in df.columns else None
    if nino12 is not None:
        kpi_card(
            "Niño 1+2 (costa do Peru)",
            f"{nino12:+.2f} degC",
            help="Aquecimento na costa peruana antecede tipicamente o "
                 "desenvolvimento de El Nino canonico via propagacao "
                 "de onda Kelvin para oeste.",
            tone="warn" if nino12 > 1.0 else "neutral",
        )

st.markdown(
    "Snapshot detalhado das 4 regioes Niño, com analise de velocidade e "
    "comparacao com transicoes historicas, esta na pagina **Estado atual**."
)

# ---------------------------------------------------------------------------
# 5. Previsao consensual CPC/IRI
# ---------------------------------------------------------------------------
st.subheader("5. Previsao consensual CPC/IRI")
st.markdown(
    "O *plume* mensal do IRI Columbia/CPC NOAA agrega previsoes de cerca de "
    "30 modelos dinamicos e estatisticos. E a referencia operacional "
    "internacional para o ENSO."
)

if not fc_off.empty:
    rows = []
    for _, row in fc_off.sort_values("target_month").iterrows():
        ph, _ = fase_para_value(row["mean"])
        prob = int(row.get("el_nino_prob", 0)) if pd.notna(row.get("el_nino_prob")) else None
        rows.append({
            "Mes alvo": row["target_month"].strftime("%b/%Y"),
            "Trimestre": row["season"],
            "SST anomalia (degC)": f"{row['mean']:+.2f}",
            "IC 90%": f"{row['q05']:+.2f} a {row['q95']:+.2f}",
            "P(El Nino)": f"{prob}%" if prob is not None else "-",
            "Categoria prevista": ph,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("Sem dados oficiais carregados. Edite `data/raw/official_forecast.csv`.")

# ---------------------------------------------------------------------------
# 6. Saida das redes neurais
# ---------------------------------------------------------------------------
st.subheader("6. Saida das redes neurais")
st.markdown(
    """
Bateria de 10 arquiteturas avaliadas em walk-forward CV (5 folds, 10 sementes
por configuracao) sobre 44 anos de dados (1982-presente):

- **Baselines**: persistencia, climatologia mensal, sazonal-naive, SARIMA, DLinear.
- **Feed-forward / recorrentes**: MLP, LSTM.
- **Modernos**: TCN (Temporal Convolutional Network), Transformer encoder-only,
  Mamba (state-space S6, implementacao puro PyTorch).

Para cada horizonte (1 a 6 meses), o vencedor por score unificado de RMSE,
R² e ACC e re-treinado com toda a serie disponivel e gera o forecast final.
Detalhamento por modelo e horizonte na pagina **Modelos**.
"""
)

if not fc_ours.empty:
    fc_sorted = fc_ours.sort_values("horizon")
    rows = []
    for _, row in fc_sorted.iterrows():
        ph, _ = fase_para_value(row["mean"])
        rows.append({
            "Mes alvo": row["target"].strftime("%b/%Y"),
            "Horizonte (meses)": int(row["horizon"]),
            "SST anomalia (degC)": f"{row['mean']:+.2f}",
            "IC 90%": f"{row['q05']:+.2f} a {row['q95']:+.2f}",
            "Modelo vencedor": row["model"],
            "N seeds": int(row["n_seeds"]),
            "Categoria prevista": ph,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info(
        "Forecast futuro ainda nao gerado. Rode `scripts/04_forecast.py` "
        "apos o training completar."
    )

# ---------------------------------------------------------------------------
# 7. Visao integrada
# ---------------------------------------------------------------------------
st.subheader("7. Visao integrada")

serie = df["nino34_anom"].dropna()
fig = go.Figure()

for nome, lo, hi, cor in FAIXAS_FASE:
    lo_v = max(lo, -3.0); hi_v = min(hi, 3.0)
    if hi_v <= lo_v: continue
    fig.add_hrect(y0=lo_v, y1=hi_v, fillcolor=cor, opacity=0.16, line_width=0)

fig.add_trace(go.Scatter(
    x=serie.index, y=serie.values, mode="lines",
    name="Niño 3.4 observada", line=dict(color="#1c2a3a", width=1.4),
))

if not fc_off.empty:
    off_sorted = fc_off.sort_values("target_month")
    fig.add_trace(go.Scatter(
        x=off_sorted["target_month"], y=off_sorted["mean"],
        mode="lines+markers", name="CPC/IRI (consensual)",
        line=dict(color="#d62728", width=2.5, dash="dot"),
        marker=dict(size=10, symbol="x"),
    ))
    if {"q05", "q95"}.issubset(off_sorted.columns):
        fig.add_trace(go.Scatter(
            x=list(off_sorted["target_month"]) + list(off_sorted["target_month"][::-1]),
            y=list(off_sorted["q95"]) + list(off_sorted["q05"][::-1]),
            fill="toself", fillcolor="rgba(214,39,40,0.16)",
            line=dict(color="rgba(0,0,0,0)"),
            name="IC CPC/IRI 90%", hoverinfo="skip",
        ))

if not fc_ours.empty:
    fc_sorted = fc_ours.sort_values("horizon")
    ref = pd.DataFrame({
        "date": [last_date] + fc_sorted["target"].tolist(),
        "mean": [last_val] + fc_sorted["mean"].tolist(),
        "q05":  [last_val] + fc_sorted["q05"].tolist(),
        "q95":  [last_val] + fc_sorted["q95"].tolist(),
    })
    fig.add_trace(go.Scatter(
        x=ref["date"], y=ref["mean"], mode="lines+markers",
        name="ensemble redes neurais",
        line=dict(color="#1f4e79", width=2.5, dash="dash"),
        marker=dict(size=10, symbol="diamond"),
    ))
    fig.add_trace(go.Scatter(
        x=list(ref["date"]) + list(ref["date"][::-1]),
        y=list(ref["q95"])  + list(ref["q05"][::-1]),
        fill="toself", fillcolor="rgba(31,78,121,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        name="IC ensemble 90%", hoverinfo="skip",
    ))

fig.update_layout(
    height=460, margin=dict(l=10, r=10, t=10, b=10),
    xaxis_title="data", yaxis_title="anomalia SST Niño 3.4 (degC)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    plot_bgcolor="white",
    font=dict(family="Inter, -apple-system, Segoe UI, Roboto, Arial, sans-serif"),
)
fig.update_yaxes(range=[-3.0, 3.0], gridcolor="#eeeeee")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.caption(
    "Linha preta: anomalia mensal observada da Niño 3.4 (1982-presente). "
    "Pontos vermelhos: previsao consensual CPC/IRI publicada em abril/2026. "
    "Pontos azuis: previsao do ensemble multi-seed das redes neurais. "
    "Faixas coloridas: limiares oficiais das categorias ENSO."
)

# ---------------------------------------------------------------------------
# 8. Estrutura do dashboard
# ---------------------------------------------------------------------------
st.subheader("8. Estrutura do dashboard")
st.markdown(
    """
- **Estado atual** - snapshot por regiao Niño, velocidade da transicao, sobreposicao
  com transicoes historicas analogas.
- **Mapa Niño** - localizacao geografica das 4 regioes monitoradas.
- **Serie historica** - serie mensal e trimestral 1950-presente, top eventos.
- **Modelos** - tabela completa de metricas por modelo x horizonte, ranking
  unificado, distribuicao das metricas, observado vs previsto.
- **Forecast** - detalhe da previsao 1-6 meses do ensemble com IC.
- **CPC/IRI vs ensemble** - comparacao quantitativa das duas fontes de previsao.
- **Contexto historico** - estatisticas descritivas dos eventos passados.
- **Auditoria de leakage** - documentacao do tratamento aplicado para evitar
  vazamento de informacao do alvo nas features.
- **Metodologia** - descricao tecnica do pipeline (ingestao, splits, modelos,
  metricas) com instrucoes de reproducao.
- **Glossario** - definicoes dos indices e siglas.
"""
)

st.caption(
    "Codigo-fonte: D:/Projetos/ENSO. Dados publicos: NOAA PSL, CPC, NCEI, SILSO. "
    "Dataset mestre regenerado mensalmente apos publicacao do plume IRI."
)
