#!/usr/bin/env pwsh
# Pipeline noturno: treina v2 com features subsuperficiais (WWV+T300),
# elege vencedores, gera forecast, converte para parquet, commita e empurra.
#
# Uso: ./scripts/07_overnight.ps1
# Estimativa total: 2.5-3.5h em RTX 3000 Ada.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot/..
$tag = "full_v2_subsuperficial"

Write-Host "[1/5] Training $tag (6 horizontes x 5 folds x 10 seeds x 10 modelos = 3000 treinos)" -ForegroundColor Cyan
uv run python scripts/03_train.py `
    --models "persistence,seasonal_naive,climatology,sarima,dlinear,mlp,lstm,tcn,transformer,mamba" `
    --horizons "1,2,3,4,5,6" --seeds "0,1,2,3,4,5,6,7,8,9" `
    --folds 5 --epochs 200 --lookback 18 --tag $tag --verbose 2>&1 |
    Tee-Object -FilePath runs/train_$tag.log

Write-Host "[2/5] Forecast 1-6m com vencedores eleitos" -ForegroundColor Cyan
uv run python scripts/04_forecast.py --train-tag $tag `
    --horizons "1,2,3,4,5,6" --seeds "0,1,2,3,4,5,6,7,8,9" `
    --epochs 200 --lookback 18 2>&1 |
    Tee-Object -FilePath runs/forecast_$tag.log

Write-Host "[3/5] Convertendo metrics e predictions para parquet (compactacao)" -ForegroundColor Cyan
uv run python -c @"
import pandas as pd, os
base = 'runs/train_$tag'
for stem in ('metrics', 'predictions'):
    csv = f'{base}/{stem}.csv'
    if os.path.exists(csv):
        kw = {'parse_dates': ['date']} if stem == 'predictions' else {}
        pd.read_csv(csv, **kw).to_parquet(f'{base}/{stem}.parquet', compression='snappy')
print('parquets gerados.')
"@

Write-Host "[4/5] Commit + push" -ForegroundColor Cyan
git add data/processed/enso_master.parquet data/processed/enso_master.csv `
        data/forecasts/forecast_future.parquet data/forecasts/forecast_future.csv `
        runs/train_$tag/metrics.parquet runs/train_$tag/predictions.parquet `
        src/enso/ingest/noaa.py src/enso/data/loader.py
git commit -m @"
data+model: re-treino v2 com features subsuperficiais (WWV + T300)

Adicionadas anomalias de Warm Water Volume (PMEL TAO) e temperatura
media 0-300m equatorial. Subsuperficie tipicamente lidera SST em
~6 meses (Meinen & McPhaden 2000), com potencial para estender skill
genuino alem do limite de h<=2 do run anterior.

- noaa.py: fetcher para wwv.dat e t300.dat (PMEL).
- loader.py: wwv_anom e t300_anom adicionados a CORE_FEATURES.
- runs/train_full_v2_subsuperficial: 3000 treinos do mesmo grid de
  10 modelos x 6 horizontes x 5 folds x 10 seeds. Vencedores eleitos
  e re-treinados para gerar forecast 1-6m.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
"@
git push

Write-Host "[5/5] Concluido." -ForegroundColor Green
Write-Host "Tag: $tag | logs em runs/train_$tag.log e runs/forecast_$tag.log"
