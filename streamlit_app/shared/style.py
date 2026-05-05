"""Paleta e estilos visuais do dashboard ENSO.

Inspirado no CABra-H: tipografia Inter, paleta azul escuro #1f4e79,
cards com sombra sutil, header h3 em destaque azul.
"""
from __future__ import annotations

from typing import Final

# Paleta principal (mesma base do CABra-H, ajustada para o tema climatico)
COR_PRIMARIA   = "#1f4e79"  # azul escuro (titulos, destaques principais)
COR_SECUNDARIA = "#3a4a5c"  # cinza-azulado (texto secundario)
COR_FUNDO_INFO = "#f5f7fa"  # caixas info
COR_BORDA      = "#cccccc"

# Paleta de fases ENSO (ordem do mais frio ao mais quente)
COR_FASE: Final[dict[str, str]] = {
    "La Nina forte":        "#08306b",
    "La Nina moderada":     "#2171b5",
    "La Nina fraca":        "#6baed6",
    "Neutro":               "#d9d9d9",
    "El Nino fraco":        "#fcae91",
    "El Nino moderado":     "#fb6a4a",
    "El Nino forte":        "#cb181d",
    "El Nino muito forte":  "#67000d",
}

# Limiares oficiais (CPC/NOAA) - SST anomaly Niño 3.4 mensal
FAIXAS_FASE: Final[list[tuple[str, float, float, str]]] = [
    ("La Nina forte",        -99.0, -1.5, COR_FASE["La Nina forte"]),
    ("La Nina moderada",      -1.5, -1.0, COR_FASE["La Nina moderada"]),
    ("La Nina fraca",         -1.0, -0.5, COR_FASE["La Nina fraca"]),
    ("Neutro",                -0.5,  0.5, COR_FASE["Neutro"]),
    ("El Nino fraco",          0.5,  1.0, COR_FASE["El Nino fraco"]),
    ("El Nino moderado",       1.0,  1.5, COR_FASE["El Nino moderado"]),
    ("El Nino forte",          1.5,  2.0, COR_FASE["El Nino forte"]),
    ("El Nino muito forte",    2.0, 99.0, COR_FASE["El Nino muito forte"]),
]


CSS_GLOBAL = """
<style>
    /* Tipografia Inter em tudo (sobrescreve serif default) */
    html, body, [class*="css"], [class*="st-emotion-cache"],
    h1, h2, h3, h4, h5, h6, p, span, a, div, li,
    [data-testid="stSidebar"] *,
    [data-testid="stSidebarNav"] *,
    [data-testid="stMarkdownContainer"] * {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }
    /* Material icons preservados */
    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    span.icon {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined",
                     "Material Icons" !important;
    }

    /* KPI cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #cccccc;
        border-radius: 6px;
        padding: 16px 18px;
        margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .kpi-label {
        font-size: 12.5px;
        color: #5a6a7a;
        margin-bottom: 4px;
        font-weight: normal;
        letter-spacing: 0.02em;
    }
    .kpi-value {
        font-size: 28px;
        color: #1c2a3a;
        font-weight: 600;
        line-height: 1.1;
    }
    .kpi-suffix {
        font-size: 14px;
        color: #5a6a7a;
        margin-left: 4px;
    }
    .kpi-good { color: #2c5e2e; }
    .kpi-warn { color: #b8702a; }
    .kpi-bad  { color: #b53a3a; }

    /* Phase badge (ENSO) */
    .phase-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 14px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    /* Section headers */
    h1, h2, h3 { font-weight: 600; color: #1c2a3a; letter-spacing: -0.01em; }
    h1 { padding-bottom: 0; margin-bottom: 4px; }
    h2 { margin-top: 18px; }
    h3 { margin-top: 14px; color: #1f4e79; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #f8f9fa; }
    [data-testid="stSidebar"] h1 {
        font-size: 18px;
        color: #1f4e79;
        border-bottom: none;
    }
    [data-testid="stSidebarNav"] ul { padding-top: 6px; }
    [data-testid="stSidebarNav"] ul li a {
        font-size: 15px !important;
        padding: 8px 14px !important;
        color: #1c2a3a !important;
    }
    [data-testid="stSidebarNav"] ul li a:hover {
        background: #e8eef5 !important;
    }

    /* Resultado destacado (caixa hero estilo CABra-H) */
    .resultado-box {
        background: linear-gradient(120deg, #1f4e79 0%, #2c5e92 100%);
        padding: 22px 28px;
        border-radius: 8px;
        color: #ffffff;
        margin: 18px 0;
    }
    .resultado-rotulo {
        font-size: 13px;
        letter-spacing: 0.06em;
        opacity: 0.85;
        text-transform: uppercase;
    }
    .resultado-valor {
        font-size: 32px;
        font-weight: 600;
        margin-top: 6px;
        margin-bottom: 4px;
    }
    .resultado-detalhe {
        font-size: 15px;
        opacity: 0.92;
    }
</style>
"""


def fase_badge_html(value: float) -> str:
    """Retorna HTML de badge colorido do estado ENSO para um valor de anomalia."""
    label = "Indefinido"
    color = "#999999"
    for nome, lo, hi, c in FAIXAS_FASE:
        if lo <= value < hi:
            label, color = nome, c
            break
    txt = _text_color_for_bg(color)
    sinal = "+" if value > 0 else ("-" if value < 0 else "")
    return (
        f'<span class="phase-badge" style="background:{color}; color:{txt};">'
        f'{label} ({sinal}{abs(value):.2f} degC)</span>'
    )


def _text_color_for_bg(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0
    return "#222" if lum > 0.6 else "white"


def fase_para_value(value: float) -> tuple[str, str]:
    for nome, lo, hi, c in FAIXAS_FASE:
        if lo <= value < hi:
            return nome, c
    return "Indefinido", "#999999"
