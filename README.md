# ENSO Forecast

Previsão multi-horizonte (1 a 6 meses) do El Niño–Oscilação Sul com auditoria de
leakage, multi-seed (10 sementes), walk-forward CV (5 folds) e bateria de 10
arquiteturas (baselines triviais, SARIMA, DLinear, MLP, LSTM, TCN, Transformer
encoder-only e Mamba puro PyTorch). Comparado lado a lado com a previsão
consensual CPC/IRI.

Dashboard companheiro do trabalho publicado em *Scientia Plena* 22 (2026).

## Setup local

Para apenas rodar o dashboard (sem retreinar modelos):

```powershell
uv sync
uv run streamlit run streamlit_app/Inicio.py
```

Para reproduzir o pipeline completo (ingestão → training → forecast):

```powershell
uv add torch torchvision --index pytorch-cu124    # GPU CUDA 12.4
uv run python scripts/01_ingest.py                # baixa dados NOAA/CPC/SILSO
uv run python scripts/02_audit_leakage.py         # auditoria de splits e features
uv run python scripts/03_train.py --tag full_v1   # treina 10 modelos x 6 horizontes
uv run python scripts/04_forecast.py --train-tag full_v1   # forecast 1-6m
uv run streamlit run streamlit_app/Inicio.py
```

## Estrutura

```
src/enso/
  ingest/    # NOAA PSL, CPC, NCEI ERSSTv5, SILSO
  data/      # loader, splits walk-forward, auditoria de leakage
  features/  # lags, rolling stats, encoding sazonal
  models/    # baselines + DL (MLP, LSTM, TCN, Transformer, Mamba)
  train/     # trainer multi-seed + walk-forward CV
  eval/      # MAE/RMSE/R²/ACC + Diebold-Mariano + score unificado
  forecast/  # ensemble multi-seed para forecast futuro
streamlit_app/
  Inicio.py  # entry point
  pages/     # 10 páginas (Estado atual, Modelos, Forecast, etc.)
  shared/    # componentes UI e estilo
data/
  raw/       # ingestão (gitignored exceto official_forecast.csv)
  processed/ # enso_master.parquet (commitado)
  forecasts/ # forecast_future.parquet (commitado)
```

## Por que esse projeto existe

O alvo natural da previsão de ENSO — o ONI — é uma média móvel centrada de 3
meses da anomalia de SST na Niño 3.4. Usar `ONI_lag1` para prever `ONI(t)`
compartilha 2 dos 3 componentes (overlap estrutural). Este projeto:

- usa **anomalia mensal de SST na Niño 3.4** como alvo (sem média móvel),
- aplica `MinMaxScaler` e seleção de features **dentro de cada fold** (walk-forward),
- reporta resultados como média ± desvio padrão de 10 sementes,
- compara com previsão consensual CPC/IRI publicada mensalmente.

A página **Auditoria de leakage** do dashboard documenta o overlap quantitativamente.

## Fontes de dados

| Índice | Fonte | Cobertura |
|---|---|---|
| Niño 1+2/3/3.4/4 SST e anomalia | CPC `sstoi.indices` | 1982-presente |
| ONI | NOAA PSL `oni.data` | 1950-presente |
| Niño 3.4 detrended | CPC `detrend.nino34.ascii` | 1950-presente |
| SOI, MEI v2, OLR, PNA, IOD, QBO, TNI | NOAA PSL | 1950-presente |
| PDO | NCEI ERSSTv5 | 1854-presente |
| Sunspots | SILSO Bélgica | 1749-presente |

Todas baixadas direto da fonte primária em `scripts/01_ingest.py`.

## Licença

Código sob MIT. Dados sob as licenças das fontes originais (NOAA: domínio público;
SILSO: CC BY-NC).
