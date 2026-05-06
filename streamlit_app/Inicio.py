"""ENSO Forecast - pagina inicial de comunicacao publica.

Versao enxuta, didatica, com previsao de curto prazo (h=1 a 3) e contexto
historico. Metodologia tecnica esta na pagina 'Visao Geral'.
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

from shared.components import aplicar_estilo, kpi_card
from shared.style import FAIXAS_FASE, fase_para_value
from utils import load_forecast, load_master

sys.path.insert(0, str(APP_DIR.parent / "src"))
from enso.forecast.official import load_official_forecast  # noqa: E402

st.set_page_config(
    page_title="Vai ter El Nino em 2026?",
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
        "<small>Esta pagina apresenta o estado e a perspectiva de curto prazo. "
        "Para metodologia, comparacao de modelos e auditoria de leakage, "
        "veja **Visao Geral** e demais paginas.</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Carrega dados
# ---------------------------------------------------------------------------
df = load_master()
fc_ours = load_forecast()
fc_off = load_official_forecast()

last_date = df["nino34_anom"].dropna().index[-1]
last_val = float(df["nino34_anom"].dropna().iloc[-1])
fase_atual, _ = fase_para_value(last_val)

fc_short = fc_ours[fc_ours["horizon"] <= 3].sort_values("horizon") if not fc_ours.empty else pd.DataFrame()
pico_oficial = fc_off["mean"].max() if not fc_off.empty else None

# ---------------------------------------------------------------------------
# Hero - pergunta direta
# ---------------------------------------------------------------------------
st.markdown(
    """
# Vai ter El Nino em 2026? Vai ser um *super*?
""",
    unsafe_allow_html=True,
)

# Resposta sintetica baseada nos dados
if pico_oficial is not None and not fc_short.empty:
    pico_nosso_3m = float(fc_short["mean"].max())
    resposta_curta = (
        f"Sim, o Pacifico equatorial deve passar para fase **El Nino** ainda em 2026. "
        f"O cenario mais provavel e **El Nino moderado a forte** (pico previsto pela "
        f"referencia internacional CPC/IRI: **+{pico_oficial:.2f} degC**, em "
        f"setembro-outubro). **Super El Nino** (acima de +2,0 degC) e **possivel mas "
        f"improvavel** - cerca de **1 em 4 chance**, segundo o boletim oficial."
    )
else:
    resposta_curta = (
        "Sim, indicios de transicao para El Nino. Atualizando previsao com novos dados."
    )

st.markdown(
    f"<div style='background:#f0f4f8; border-left:4px solid #1f4e79; "
    f"padding:14px 18px; border-radius:4px; font-size:16px; line-height:1.55; "
    f"margin:8px 0 24px 0;'>{resposta_curta}</div>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tres cards
# ---------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("##### Onde estamos agora")
    kpi_card(
        f"Niño 3.4 ({last_date:%b/%Y})",
        f"{last_val:+.2f} degC",
        help=f"Categoria: {fase_atual}",
        tone="warn" if last_val > 0.5 else "neutral",
    )
    st.caption(
        f"Anomalia mensal de temperatura na regiao do Pacifico central usada "
        f"como referencia oficial para o ENSO. Valor atual ainda dentro da faixa "
        f"**neutra** (-0,5 a +0,5 degC), mas com tendencia de subida desde o "
        f"final de 2025."
    )

with c2:
    st.markdown("##### Pra onde vamos (proximos 3 meses)")
    if not fc_short.empty:
        ult = fc_short.iloc[-1]
        ph, _ = fase_para_value(float(ult["mean"]))
        kpi_card(
            f"{ult['target']:%b/%Y} (h={int(ult['horizon'])})",
            f"{float(ult['mean']):+.2f} degC",
            help=f"Modelo: {ult['model']}",
            tone="warn" if ult["mean"] > 0.5 else "neutral",
        )
        st.caption(
            f"Previsao do nosso ensemble de redes neurais (vencedor: "
            f"**{ult['model']}**) para **{ult['target']:%b/%Y}**: {ph}. "
            f"Concordancia com o consenso CPC/IRI dentro de 0,2 degC ate o "
            f"horizonte de 2 meses."
        )
    else:
        st.info("Forecast em atualizacao.")

with c3:
    st.markdown("##### E o pior da historia?")
    if pico_oficial is not None:
        kpi_card(
            "Pico previsto (oficial)",
            f"+{pico_oficial:.2f} degC",
            suffix="< +2,00",
            help="Limiar de 'super El Nino' = +2,0 degC",
            tone="neutral",
        )
        st.caption(
            f"Os tres super El Ninos historicos (1982-83, 1997-98, 2015-16) "
            f"tiveram pico **acima de +2,5 degC**. O cenario central previsto para "
            f"2026 fica **{2.0 - pico_oficial:.2f} degC abaixo** do limiar de "
            f"super, com IC superior do plume tocando +2,4."
        )

st.divider()

# ---------------------------------------------------------------------------
# Grafico principal: ultimos 36 meses + previsao 1-3
# ---------------------------------------------------------------------------
st.subheader("Como a temperatura do Pacifico esta mudando")
st.caption(
    "Cada ponto preto e a anomalia de temperatura do mar na regiao Niño 3.4 "
    "(Pacifico central) num mes. Acima de +0,5 degC sao condicoes de El Nino; "
    "abaixo de -0,5 degC, La Nina. Os pontos azuis sao a previsao do nosso "
    "ensemble de redes neurais para os proximos 3 meses (horizontes em que o "
    "modelo tem skill estatistico validado)."
)

serie = df["nino34_anom"].dropna().tail(36)
fig = go.Figure()

for nome, lo, hi, cor in FAIXAS_FASE:
    lo_v = max(lo, -3.0); hi_v = min(hi, 3.0)
    if hi_v <= lo_v: continue
    fig.add_hrect(y0=lo_v, y1=hi_v, fillcolor=cor, opacity=0.18, line_width=0)

fig.add_trace(go.Scatter(
    x=serie.index, y=serie.values, mode="lines+markers",
    name="observado",
    line=dict(color="#1c2a3a", width=2.2),
    marker=dict(size=6),
))

if not fc_short.empty:
    ref = pd.DataFrame({
        "date": [last_date] + fc_short["target"].tolist(),
        "mean": [last_val] + fc_short["mean"].tolist(),
        "q05":  [last_val] + fc_short["q05"].tolist(),
        "q95":  [last_val] + fc_short["q95"].tolist(),
    })
    fig.add_trace(go.Scatter(
        x=ref["date"], y=ref["mean"], mode="lines+markers",
        name="previsao (nosso ensemble)",
        line=dict(color="#1f4e79", width=2.5, dash="dash"),
        marker=dict(size=10, symbol="diamond"),
    ))
    fig.add_trace(go.Scatter(
        x=list(ref["date"]) + list(ref["date"][::-1]),
        y=list(ref["q95"])  + list(ref["q05"][::-1]),
        fill="toself", fillcolor="rgba(31,78,121,0.20)",
        line=dict(color="rgba(0,0,0,0)"),
        name="intervalo de confianca 90%", hoverinfo="skip",
    ))

if not fc_off.empty:
    off_short = fc_off[fc_off["target_month"] <= (last_date + pd.DateOffset(months=4))]
    if not off_short.empty:
        fig.add_trace(go.Scatter(
            x=off_short["target_month"], y=off_short["mean"],
            mode="lines+markers", name="referencia internacional CPC/IRI",
            line=dict(color="#d62728", width=2, dash="dot"),
            marker=dict(size=9, symbol="x"),
        ))

fig.update_layout(
    height=420, plot_bgcolor="white",
    xaxis_title="data", yaxis_title="anomalia de temperatura (degC)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    font=dict(family="Inter, -apple-system, Segoe UI, Arial, sans-serif"),
    margin=dict(l=10, r=10, t=10, b=10),
)
fig.update_yaxes(range=[-2.5, 2.5], gridcolor="#eeeeee")
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.divider()

# ---------------------------------------------------------------------------
# O que isso significa pra cada regiao do Brasil
# ---------------------------------------------------------------------------
st.subheader("O que um El Nino moderado a forte costuma significar para o Brasil")
st.caption(
    "Padroes de teleconexao mais frequentemente reportados na literatura para "
    "fases positivas do ENSO. Sao **probabilidades aumentadas**, nao certezas: "
    "eventos individuais dependem de outros fatores (frentes frias, MJO, "
    "Dipolo do Indico, processos locais)."
)

reg1, reg2 = st.columns(2)
with reg1:
    st.markdown(
        """
**Sul do Brasil** (RS, SC, PR)
Probabilidade aumentada de **chuvas acima da media** entre primavera e
verao, com risco de enchentes em areas vulneraveis e impacto na safra
de inverno (trigo).

**Sudeste e Centro-Oeste**
Tendencia de **inverno mais seco e quente** que a media, com efeitos no
nivel de reservatorios hidreletricos e no plantio da safra de verao.
        """
    )
with reg2:
    st.markdown(
        """
**Nordeste (sertao semiarido)**
Risco de **chuvas abaixo da media** durante a estacao chuvosa (fev-mai
de 2027), com impacto sobre seguranca hidrica e producao agropecuaria.

**Norte (Amazonia leste)**
Tendencia de **estiagem mais prolongada**, com aumento do risco de
incendios florestais ja a partir do segundo semestre de 2026.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Como entender essa previsao
# ---------------------------------------------------------------------------
st.subheader("Como interpretar essa previsao")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown(
        """
**A previsao tem incerteza, e isso e proprio do sistema.**
O ENSO e uma oscilacao acoplada oceano-atmosfera com componente
parcialmente caotico. Modelos atuais (incluindo o nosso) tem **skill
util ate cerca de 3 meses** com a configuracao estudada. Alem disso,
a incerteza cresce rapidamente.

**Por que confiamos no que mostramos para mai-jul/2026:**
o ensemble foi avaliado em 5 janelas historicas distintas (2000-02,
2006-08, 2012-14, 2017-19 e 2023-25), com R² medio positivo nesses
horizontes (0.69, 0.44 e 0.19 para 1, 2 e 3 meses, respectivamente).
        """
    )
with col_b:
    st.markdown(
        """
**Por que "super El Nino" nao deve dominar a leitura:**
o pico central previsto pelo consenso internacional (CPC/IRI) e
**+1,57 degC** em setembro-outubro - categoria El Nino **forte**, nao
super. A probabilidade reportada de o evento ultrapassar +2,0 degC
no inverno boreal de 2026-27 e **cerca de 25%** - possivel, mas
minoritario.

**O que monitorar nos proximos meses:**
calor subsuperficial do Pacifico equatorial (WWV) e propagacao da
onda Kelvin oceanica. Se ambos seguirem subindo no ritmo recente,
a probabilidade de evento mais intenso aumenta.
        """
    )

st.divider()

# ---------------------------------------------------------------------------
# Para aprofundar
# ---------------------------------------------------------------------------
st.subheader("Para aprofundar")
st.markdown(
    """
- **Visao Geral**: descricao tecnica do fenomeno, variaveis monitoradas e
  estado atual em detalhe.
- **Estado atual**: snapshot por regiao do Pacifico (Niño 1+2, 3, 3.4, 4),
  velocidade da transicao recente e comparacao com episodios historicos analogos.
- **Forecast**: previsao detalhada 1-6 meses do ensemble, com intervalos de
  confianca e indicacao de skill por horizonte.
- **Oficial vs modelo**: comparacao quantitativa entre o nosso ensemble e o
  consenso CPC/IRI (cerca de 30 modelos dinamicos e estatisticos).
- **Auditoria de leakage**: por que o alvo foi a anomalia mensal de SST e
  nao o ONI; demonstracao quantitativa do vies estrutural evitado.
- **Metodologia**: pipeline completo (ingestao, splits walk-forward,
  10 arquiteturas comparadas, multi-seed) com instrucoes de reproducao.
- **Glossario**: definicoes dos indices climaticos e siglas usadas no app.
"""
)

st.caption(
    "Fontes de dados: NOAA PSL, CPC, NCEI ERSSTv5, SILSO, PMEL/TAO. "
    "Codigo-fonte: github.com/viniciusazeved/enso-forecast. "
    "Atualizacao mensal apos publicacao do plume IRI."
)
