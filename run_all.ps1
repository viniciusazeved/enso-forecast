#!/usr/bin/env pwsh
# Pipeline end-to-end: ingestao -> auditoria -> training -> forecast -> dashboard.
# Uso: ./run_all.ps1 [-quick]

param(
    [switch]$quick = $false,
    [string]$tag = "full_v1"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "[1/5] sync deps com uv..." -ForegroundColor Cyan
uv sync

Write-Host "`n[2/5] ingestao NOAA + CPC + SILSO..." -ForegroundColor Cyan
uv run python scripts/01_ingest.py

Write-Host "`n[3/5] auditoria de leakage..." -ForegroundColor Cyan
uv run python scripts/02_audit_leakage.py

Write-Host "`n[4/5] training..." -ForegroundColor Cyan
if ($quick) {
    uv run python scripts/03_train.py --quick --models "persistence,seasonal_naive,climatology,dlinear,mlp,lstm,tcn,transformer" --tag $tag
} else {
    uv run python scripts/03_train.py --tag $tag --verbose
}

Write-Host "`n[5/5] forecast 1-6 meses..." -ForegroundColor Cyan
uv run python scripts/04_forecast.py --train_tag $tag

Write-Host "`nPronto. Para rodar o dashboard:" -ForegroundColor Green
Write-Host "  uv run streamlit run streamlit_app/app.py" -ForegroundColor Yellow
