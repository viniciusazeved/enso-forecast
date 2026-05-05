"""Glossario didatico dos indices climaticos usados no projeto."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pandas as pd
import streamlit as st

from shared.components import aplicar_estilo, header_pagina

st.set_page_config(page_title="Glossario - ENSO", page_icon="🌊", layout="wide")
aplicar_estilo()

header_pagina(
    "Glossario - indices e siglas",
    "Definicoes rapidas dos termos usados no app, organizadas por categoria.",
)

GLOSSARY = [
    {
        "sigla": "ENSO / ENOS",
        "nome":  "El Nino-Oscilacao Sul",
        "o que e": "Padrao acoplado oceano-atmosfera no Pacifico equatorial. Tem 3 fases:"
                   " El Nino (aquecimento), La Nina (resfriamento), Neutro.",
    },
    {
        "sigla": "Niño 3.4",
        "nome":  "Regiao 5N-5S, 170W-120W",
        "o que e": "Caixa do Pacifico central usada como referencia oficial do ENSO."
                   " A SST media dentro dela define o estado atual.",
    },
    {
        "sigla": "ONI",
        "nome":  "Oceanic Niño Index",
        "o que e": "Media movel **trimestral** da anomalia de SST na Niño 3.4."
                   " >=+0.5 degC por 5 trimestres consecutivos = El Nino oficial.",
    },
    {
        "sigla": "SOI",
        "nome":  "Southern Oscillation Index",
        "o que e": "Diferenca normalizada de pressao entre Tahiti e Darwin. Sinal"
                   " atmosferico do ENSO; geralmente em fase oposta ao ONI.",
    },
    {
        "sigla": "MEI v2",
        "nome":  "Multivariate ENSO Index v2",
        "o que e": "Combina 5 variaveis (SST, vento, pressao, OLR, etc) num unico"
                   " indice. Mais robusto que olhar so SST.",
    },
    {
        "sigla": "OLR",
        "nome":  "Outgoing Longwave Radiation",
        "o que e": "Proxy de conveccao tropical: menos OLR = mais nuvens convectivas."
                   " Em El Nino, conveccao desloca-se para o Pacifico central.",
    },
    {
        "sigla": "PNA",
        "nome":  "Pacific North American pattern",
        "o que e": "Padrao de circulacao atmosferica de medias latitudes; modulado"
                   " pelo ENSO e influencia tempo na America do Norte.",
    },
    {
        "sigla": "PDO",
        "nome":  "Pacific Decadal Oscillation",
        "o que e": "Variabilidade decadal do Pacifico. Modula a *frequencia* dos"
                   " eventos ENSO, mas nao causa eventos individuais.",
    },
    {
        "sigla": "AMO",
        "nome":  "Atlantic Multidecadal Oscillation",
        "o que e": "Variabilidade multidecadal da SST do Atlantico Norte. Pode"
                   " influenciar ENSO via teleconexoes interbacias.",
    },
    {
        "sigla": "IOD",
        "nome":  "Indian Ocean Dipole",
        "o que e": "Gradiente de SST oeste/leste do Indico. Interage com ENSO,"
                   " amplificando ou amortecendo.",
    },
    {
        "sigla": "QBO",
        "nome":  "Quasi-Biennial Oscillation",
        "o que e": "Oscilacao do vento zonal estratosferico tropical (~28 meses)."
                   " Pode modular previsibilidade do ENSO.",
    },
    {
        "sigla": "TNI",
        "nome":  "Trans-Niño Index",
        "o que e": "Diferenca entre anomalia em Niño 1+2 e Niño 4. Captura o"
                   " gradiente leste-oeste, util para tipo de El Nino (canonico vs Modoki)."
    },
    {
        "sigla": "TSI / sunspots",
        "nome":  "Total Solar Irradiance / manchas solares",
        "o que e": "Atividade solar. Modulacao de longo prazo no balanco energetico"
                   " - efeito sobre ENSO ainda em estudo.",
    },
    {
        "sigla": "Walk-forward CV",
        "nome":  "Validacao cruzada com janelas expansivas",
        "o que e": "Treina sempre no passado, valida no futuro. Sem embaralhamento."
                   " Padrao-ouro para series temporais.",
    },
    {
        "sigla": "ACC",
        "nome":  "Anomaly Correlation Coefficient",
        "o que e": "Pearson sobre anomalias. Metrica padrao em previsao climatica;"
                   " mede se modelo acerta a forma da curva (nao so o nivel)."
    },
    {
        "sigla": "Persistence",
        "nome":  "Baseline ingenua",
        "o que e": "Previsao = ultima observacao. Para horizonte 1, e dificil de bater."
                   " Modelo que nao bate persistence so memorizou autocorrelacao."
    },
]

st.dataframe(pd.DataFrame(GLOSSARY), use_container_width=True, hide_index=True)
