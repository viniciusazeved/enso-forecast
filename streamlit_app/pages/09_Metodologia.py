"""Metodologia tecnica resumida para reprodutibilidade e curiosos."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st

from shared.components import aplicar_estilo, header_pagina

st.set_page_config(page_title="Metodologia - ENSO", page_icon="🌊", layout="wide")
aplicar_estilo()

header_pagina(
    "Metodologia em uma pagina",
    "Descricao tecnica do pipeline (ingestao, features, splits, modelos, metricas) "
    "com instrucoes de reproducao end-to-end.",
)

st.subheader("1. Fontes de dados")
st.markdown(
    """
    Todas as series sao publicas, mensais, baixadas direto da fonte primaria:

    | Indice | Fonte | Cobertura |
    |--------|-------|-----------|
    | Niño 1+2/3/3.4/4 SST e anomalia | CPC sstoi.indices | 1982-presente |
    | ONI (referencia oficial) | NOAA PSL oni.data | 1950-presente |
    | Niño 3.4 longa | CPC detrend.nino34.ascii | 1950-presente |
    | SOI, PNA, IOD, QBO, TNI | NOAA PSL | 1950-presente |
    | MEI v2 | NOAA PSL meiv2.data | 1979-presente |
    | OLR | NOAA PSL olr.data | 1974-presente |
    | PDO | NCEI ERSSTv5 | 1854-presente |
    | AMO | NOAA PSL amon.us | 1950-2023 |
    | Sunspots | SILSO Belgica | 1749-presente |

    Ingestor: `src/enso/ingest/noaa.py`. Roda via `uv run python scripts/01_ingest.py`.
    """
)

st.subheader("2. Alvo, features e auditoria de leakage")
st.markdown(
    """
    - **Alvo**: `nino34_anom` (anomalia mensal de SST na Niño 3.4, do CPC sstoi).
      *Nao* ONI - para evitar overlap estrutural da media movel centrada.
    - **Features removidas**: `oni`, `nino34_sst`, `nino34_anom_long` (colineares).
    - **Lookback**: 18 meses de janela para modelos sequenciais.
    - **Splits**: walk-forward expanding window (5 folds), validacao = 24 meses,
      teste = 24 meses, treino minimo = 180 meses.

    Auditorias automaticas em `src/enso/data/leakage.py`:
    1. monotonicidade temporal (treino < val < teste, sem overlap),
    2. scaler refitado por fold,
    3. alvo nao esta nas features,
    4. nenhuma feature com corr ~= 1 com alvo no instante t,
    5. todo lag k >= 1 (sem leitura do futuro).

    Falha em qualquer um para o pipeline com `LeakageError`.
    """
)

st.subheader("3. Modelos avaliados")
st.markdown(
    """
    | Categoria | Modelos |
    |-----------|---------|
    | Baselines triviais | Persistence, Seasonal Naive, Climatology |
    | Estatisticos | SARIMA(1,0,1)(1,0,1,12) |
    | Lineares fortes | DLinear (Zeng et al. 2022) |
    | Redes feed-forward | MLP |
    | Recorrentes | LSTM (2 camadas, 64 hidden) |
    | Convolucionais | TCN (3 blocos dilated, kernel 3) |
    | Atencao | Transformer (encoder-only, 2 camadas, 4 heads) |
    | State-Space | Mamba/S6 (puro PyTorch, scan recursivo) |

    Codigo dos modelos em `src/enso/models/`. Interface comum em `base.py`.
    """
)

st.subheader("4. Treinamento e avaliacao")
st.markdown(
    """
    - Otimizador Adam (lr=1e-3, weight_decay=1e-5), MSE loss, gradient clipping em 1.0.
    - Early stopping com paciencia de 25 epocas sobre val_loss.
    - **10 sementes** por configuracao -> media +/- desvio reportados.
    - **5 folds walk-forward** -> robustez temporal.
    - Total: 6 horizontes x 5 folds x 10 seeds x 10 modelos = **3000 treinos por run**.

    Metricas em `src/enso/eval/metrics.py`:
    - **MAE, RMSE**: erros absolutos (escala da serie).
    - **R^2**: variancia explicada (cuidado: pode ser negativo).
    - **ACC**: anomaly correlation, padrao em previsao climatica.
    - **Hit rate** de fase: concordancia La Nina/Neutro/El Nino com limiar +/-0.5.
    - **Diebold-Mariano**: teste de igualdade de acuracia preditiva entre modelos
      (`src/enso/eval/compare.py`).

    Score unificado por horizonte normaliza (0..1) MAE, RMSE, R^2, ACC e tira media simples.
    """
)

st.subheader("5. Forecasting futuro")
st.markdown(
    """
    Apos eleger o vencedor por horizonte, re-treinamos com **todo o historico
    disponivel** (treino + val do walk-forward). Geramos previsao para os
    proximos 1-6 meses com **ensemble multi-seed**: media e quantis 5-25-50-75-95
    sobre as 10 sementes.

    O intervalo de confianca (q05 a q95) sumariza a incerteza epistemica do modelo.
    """
)

st.subheader("6. Comparacao com referencia oficial")
st.markdown(
    """
    A previsao consensual do **IRI/CPC** (publicada mensalmente no plume) entra
    como referencia em `data/raw/official_forecast.csv`. Para atualizar a cada mes:

    1. abra https://iri.columbia.edu/our-expertise/climate/forecasts/enso/current/,
    2. copie a tabela de medias previstas + probabilidades,
    3. edite o CSV (formato: `target_month,season,mean,q05,q95,el_nino_prob,...`),
    4. recarregue o app (botao 'rerun' do Streamlit).

    O IC do oficial vem do plume IRI (espalhamento entre os modelos do consorcio).
    """
)

st.subheader("7. Como rodar o projeto inteiro")
st.code(
    """
    # 1. setup
    uv sync
    uv add torch torchvision

    # 2. ingestao (1-2 min)
    uv run python scripts/01_ingest.py

    # 3. auditoria de leakage (segundos)
    uv run python scripts/02_audit_leakage.py

    # 4. training completo (~2h em RTX 3000 Ada)
    uv run python scripts/03_train.py --tag full_v1 --verbose

    # 5. forecast 1-6 meses (~10 min)
    uv run python scripts/04_forecast.py --train_tag full_v1

    # 6. dashboard
    uv run streamlit run streamlit_app/app.py
    """,
    language="powershell",
)
