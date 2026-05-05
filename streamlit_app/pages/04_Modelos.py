"""Comparacao de modelos: tabela mestre, boxplot por horizonte, ranking unificado."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from enso.config import RUNS_DIR  # noqa: E402
from enso.eval.compare import unified_score, winner_per_horizon  # noqa: E402

from shared.components import aplicar_estilo, header_pagina
from utils import load_metrics, load_predictions

st.set_page_config(page_title="Modelos - ENSO", page_icon="🌊", layout="wide")
aplicar_estilo()

header_pagina(
    "Comparacao de modelos",
    "Resultados do walk-forward CV (5 folds) com 10 sementes por configuracao. "
    "Metricas reportadas como media +/- desvio.",
)

# ---------------------------------------------------------------------------
# 1. Como ler esta pagina
# ---------------------------------------------------------------------------
st.subheader("1. Configuracao experimental")
st.markdown(
    """
- **10 arquiteturas** comparadas: persistencia, sazonal-naive, climatologia,
  SARIMA(1,0,1)(1,0,1,12), DLinear, MLP, LSTM, TCN, Transformer encoder-only,
  Mamba (S6 puro PyTorch).
- **6 horizontes**: 1, 2, 3, 4, 5 e 6 meses.
- **5 folds walk-forward** com janela expansiva (treino minimo 180 meses,
  validacao 24 meses, teste 24 meses).
- **10 sementes aleatorias** por configuracao.
- Para cada combinacao (modelo x horizonte), as metricas sao reportadas como
  **media +/- desvio padrao** sobre os 50 treinos (5 folds x 10 seeds).
"""
)

# Detecta runs disponiveis automaticamente. v2 (com WWV/T300) eh o default.
_PRIORITY = {"full_v2_subsuperficial": 0, "full_v1_completo": 1}
available = sorted(
    [p.name.replace("train_", "") for p in RUNS_DIR.glob("train_*")
     if (p / "metrics.parquet").exists() or (p / "metrics.csv").exists()],
    key=lambda x: (_PRIORITY.get(x, 99), x),
)
if not available:
    st.warning("Nenhum run encontrado. Rode `scripts/03_train.py`.")
    st.stop()

tag = st.selectbox("Run", available, index=0)
df_m = load_metrics(tag)
df_p = load_predictions(tag)

if df_m.empty:
    st.warning(f"Sem metricas para a tag '{tag}'.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# 2. Vencedor por horizonte
# ---------------------------------------------------------------------------
st.subheader("2. Vencedor por horizonte")
st.markdown(
    "Score unificado normaliza RMSE, R² e ACC entre os modelos no mesmo "
    "horizonte e tira a media. Quanto maior, melhor o desempenho relativo."
)

score = unified_score(df_m).round(3)
winners = winner_per_horizon(df_m).round(3)

left, right = st.columns([2, 1])
with left:
    fig_score = px.bar(
        score.sort_values(["horizon", "score"], ascending=[True, False]),
        x="horizon", y="score", color="model", barmode="group",
        title="Score unificado (0 a 1; maior = melhor)",
        labels={"horizon": "horizonte (meses)", "score": "score unificado"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_score.update_layout(
        height=400, plot_bgcolor="white",
        font=dict(family="Inter, -apple-system, Segoe UI, Arial, sans-serif"),
    )
    st.plotly_chart(fig_score, use_container_width=True)
with right:
    st.markdown("**Vencedor por horizonte**")
    st.dataframe(
        winners[["horizon", "model", "score", "rmse", "r2", "acc"]].reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# 3. Resumo agregado
# ---------------------------------------------------------------------------
st.subheader("3. Resumo agregado")
agg = (
    df_m.groupby(["model", "horizon"])[["mae", "rmse", "r2", "acc", "hit"]]
    .agg(["mean", "std"]).round(3)
)
st.dataframe(agg, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 4. Distribuicao das metricas
# ---------------------------------------------------------------------------
st.subheader("4. Distribuicao das metricas por modelo")
metric_choice = st.selectbox("Metrica", ["rmse", "r2", "acc"], index=0)
fig_box = px.box(
    df_m, x="model", y=metric_choice, color="model",
    facet_col="horizon", facet_col_wrap=3, points="all",
    labels={"model": "modelo"},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_box.update_layout(
    height=600, showlegend=False, plot_bgcolor="white",
    font=dict(family="Inter, -apple-system, Segoe UI, Arial, sans-serif"),
)
st.plotly_chart(fig_box, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# 5. Curvas observado vs previsto
# ---------------------------------------------------------------------------
st.subheader("5. Observado vs previsto, por modelo")
if not df_p.empty:
    h_choice = st.selectbox("Horizonte", sorted(df_p["horizon"].unique().tolist()))
    sub = df_p[df_p["horizon"] == h_choice]
    avg = sub.groupby(["model", "date"])[["y_true", "y_pred"]].mean().reset_index()
    fig_pred = go.Figure()
    truth = avg.groupby("date")["y_true"].first().sort_index()
    fig_pred.add_trace(go.Scatter(
        x=truth.index, y=truth.values, mode="lines", name="observado",
        line=dict(color="black", width=2.2),
    ))
    palette = px.colors.qualitative.Bold
    for i, (model, g) in enumerate(avg.groupby("model")):
        fig_pred.add_trace(go.Scatter(
            x=g["date"], y=g["y_pred"], mode="lines", name=model,
            line=dict(color=palette[i % len(palette)], width=1.4, dash="dash"),
        ))
    fig_pred.update_layout(
        height=440, plot_bgcolor="white",
        xaxis_title="data", yaxis_title="anomalia Niño 3.4 (degC)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="Inter, -apple-system, Segoe UI, Arial, sans-serif"),
    )
    st.plotly_chart(fig_pred, use_container_width=True)
