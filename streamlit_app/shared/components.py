"""Componentes UI reutilizaveis do dashboard ENSO."""
from __future__ import annotations

from typing import Any

import streamlit as st

from shared.style import CSS_GLOBAL, fase_badge_html, fase_para_value


def aplicar_estilo() -> None:
    """Injeta o CSS global."""
    st.markdown(CSS_GLOBAL, unsafe_allow_html=True)


def kpi_card(
    label: str,
    value: str | int | float,
    suffix: str = "",
    *,
    tone: str = "neutral",
    help: str | None = None,
) -> None:
    """Card de KPI com tipografia consistente.

    tone: 'neutral' | 'good' | 'warn' | 'bad'
    """
    tone_class = {"good": "kpi-good", "warn": "kpi-warn", "bad": "kpi-bad"}.get(tone, "")
    if isinstance(value, (int, float)):
        if isinstance(value, int) or value == int(value):
            value_fmt = f"{int(value):,}".replace(",", ".")
        else:
            value_fmt = f"{value:+.2f}"
    else:
        value_fmt = str(value)

    suffix_html = f'<span class="kpi-suffix">{suffix}</span>' if suffix else ""
    help_attr = f' title="{help}"' if help else ""

    html = f"""
    <div class="kpi-card"{help_attr}>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value {tone_class}">{value_fmt}{suffix_html}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def fase_badge(value: float) -> None:
    st.markdown(fase_badge_html(value), unsafe_allow_html=True)


def resultado_destaque(rotulo: str, valor: str, detalhe: str) -> None:
    """Caixa hero com gradiente azul (estilo CABra-H 'Resultado da Fase 0')."""
    st.markdown(
        f"""
        <div class='resultado-box'>
            <div class='resultado-rotulo'>{rotulo}</div>
            <div class='resultado-valor'>{valor}</div>
            <div class='resultado-detalhe'>{detalhe}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def header_pagina(titulo: str, subtitulo: str | None = None) -> None:
    """Cabecalho padrao de pagina com titulo e subtitulo opcional."""
    st.markdown(f"# {titulo}")
    if subtitulo:
        st.markdown(
            f"<p style='color:#5a6a7a; font-size:15px; margin-top:0; "
            f"margin-bottom:18px; line-height:1.5;'>{subtitulo}</p>",
            unsafe_allow_html=True,
        )
