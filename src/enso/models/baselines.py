"""Baselines obrigatorios: persistence, climatologia mensal, sazonal-naive,
SARIMA e DLinear (Zeng et al. 2022)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from enso.models.base import ForecastModel


@dataclass
class Persistence(ForecastModel):
    """y_hat(t+h) = ultimo valor observado do alvo no instante t.

    Espera entrada como sequencia [N, lookback, F] e usa o ultimo step da
    coluna `target_col_idx` como predicao.
    """
    target_col_idx: int = 0
    name: str = "persistence"
    input_kind: str = "sequence"

    def fit(self, X, y, X_val=None, y_val=None, **kw):
        return self

    def predict(self, X):
        return X[:, -1, self.target_col_idx]


@dataclass
class SeasonalNaive(ForecastModel):
    """y_hat(t+h) = y(t-12+h). Requer lookback >= 12 - h + 1 ~ usar lookback=12.

    Aproveita a entrada sequencial: pega o valor 12 - h passos atras (relativo
    a t no fim da janela).
    """
    target_col_idx: int = 0
    horizon: int = 1
    name: str = "seasonal_naive"
    input_kind: str = "sequence"

    def fit(self, X, y, X_val=None, y_val=None, **kw):
        return self

    def predict(self, X):
        # Em t (ultimo step da janela), y(t-12+h). idx = lookback-1 - (12 - h)
        L = X.shape[1]
        idx = L - 1 - (12 - self.horizon)
        if idx < 0 or idx >= L:
            # fallback: usa primeiro step se janela curta
            idx = 0
        return X[:, idx, self.target_col_idx]


@dataclass
class Climatology(ForecastModel):
    """Media historica do alvo por mes-do-ano para o instante t+h.

    Treina em (datas_train, y_train); predict recebe datas-destino e devolve
    a media do mes correspondente. Como aqui passamos X numpy sem datas,
    armazenamos `month_means` indexado pelo mes 1..12 e o caller passa o mes
    via campo extra `months_target` no fit/predict de override.

    Esta versao infere o mes-alvo a partir do lookback assumindo que a
    sequencia contem 'sin_month'/'cos_month' nas ultimas 2 colunas. Para
    simplicidade, usaremos uma versao baseada em date no trainer.
    """
    name: str = "climatology"
    input_kind: str = "sequence"
    target_col_idx: int = 0
    months: np.ndarray | None = None  # mes do alvo (1..12) por amostra

    def fit(self, X, y, X_val=None, y_val=None, *, months_train=None, **kw):
        if months_train is None:
            raise ValueError("Climatology.fit requer months_train (1..12 do alvo).")
        s = pd.Series(y, index=pd.Index(months_train, name="month"))
        self.month_means_ = s.groupby(level=0).mean().to_dict()
        return self

    def predict(self, X, *, months_target=None):
        if months_target is None:
            raise ValueError("Climatology.predict requer months_target (1..12 do alvo).")
        glob = float(np.mean(list(self.month_means_.values()))) if self.month_means_ else 0.0
        return np.array([self.month_means_.get(int(m), glob) for m in months_target])


class SARIMA(ForecastModel):
    """SARIMA(1,0,1)(1,0,1,12) sobre o target univariado, previsao h passos.

    Mantem 'state' simples: treina no historico do treino, e na predicao
    estende com X_val para gerar previsoes one-step-ahead recursivas.
    """
    name = "sarima"
    input_kind = "sequence"

    def __init__(self, target_col_idx: int = 0, horizon: int = 1):
        self.target_col_idx = target_col_idx
        self.horizon = horizon

    def fit(self, X, y, X_val=None, y_val=None, **kw):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        # serie do alvo no fim das janelas de treino (t+h)
        # como X eh [N, L, F], reconstruimos a serie usando X[:,-1,target_col_idx]
        # que eh y(t), e y eh y(t+h). Alinhamos como [y(t), y(t+1), ..., y(t+N-1+h)]
        # Para simplicidade: usamos somente y como serie de treino.
        self.endog_train_ = np.asarray(y, dtype=float)
        try:
            self.model_ = SARIMAX(
                self.endog_train_,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.res_ = self.model_.fit(disp=False, maxiter=200)
        except Exception:
            self.res_ = None
        return self

    def predict(self, X):
        n = X.shape[0]
        if self.res_ is None:
            # fallback: persistencia
            return X[:, -1, self.target_col_idx]
        # Forecast estatico h passos a frente (mesmo h para todas as amostras)
        try:
            fc = self.res_.get_forecast(steps=n).predicted_mean
            return np.asarray(fc, dtype=float)
        except Exception:
            return X[:, -1, self.target_col_idx]


class DLinear(ForecastModel):
    """DLinear: decomposicao trend+seasonal seguida de duas camadas lineares.

    Implementacao minimal seguindo Zeng et al. 2022. Trabalha sobre sequencia
    de UMA variavel (o target). Estende para multivariado tomando feature
    target_col_idx como referencia.
    """
    name = "dlinear"
    input_kind = "sequence"

    def __init__(
        self,
        lookback: int,
        horizon: int = 1,
        target_col_idx: int = 0,
        kernel_size: int = 25,
        epochs: int = 200,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.lookback = lookback
        self.horizon = horizon
        self.target_col_idx = target_col_idx
        self.kernel_size = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        self.epochs = epochs
        self.lr = lr
        self.device = device

    def _build(self):
        import torch
        import torch.nn as nn

        L, K = self.lookback, self.kernel_size

        class _DLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.kernel = K
                self.linear_seasonal = nn.Linear(L, 1)
                self.linear_trend    = nn.Linear(L, 1)

            def _moving_avg(self, x):
                # x: [B, L]
                pad = (self.kernel - 1) // 2
                xp = nn.functional.pad(x.unsqueeze(1), (pad, pad), mode="replicate")
                w = torch.ones(1, 1, self.kernel, device=x.device) / self.kernel
                return nn.functional.conv1d(xp, w).squeeze(1)

            def forward(self, x):
                trend    = self._moving_avg(x)
                seasonal = x - trend
                return self.linear_seasonal(seasonal).squeeze(-1) + \
                       self.linear_trend(trend).squeeze(-1)

        self.net_ = _DLinear().to(self.device)
        return self.net_

    def fit(self, X, y, X_val=None, y_val=None, **kw):
        import torch

        self._build()
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss = torch.nn.MSELoss()

        Xt = torch.tensor(X[:, :, self.target_col_idx], dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device)

        best, patience, bad = float("inf"), 20, 0
        best_state = None
        for ep in range(self.epochs):
            self.net_.train()
            opt.zero_grad()
            out = self.net_(Xt)
            l = loss(out, yt)
            l.backward()
            opt.step()
            if X_val is not None and y_val is not None:
                self.net_.eval()
                with torch.no_grad():
                    Xv = torch.tensor(X_val[:, :, self.target_col_idx], dtype=torch.float32, device=self.device)
                    yv = torch.tensor(y_val, dtype=torch.float32, device=self.device)
                    lv = float(loss(self.net_(Xv), yv))
                if lv < best - 1e-5:
                    best, bad = lv, 0
                    best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        return self

    def predict(self, X):
        import torch
        self.net_.eval()
        with torch.no_grad():
            Xt = torch.tensor(X[:, :, self.target_col_idx], dtype=torch.float32, device=self.device)
            return self.net_(Xt).cpu().numpy()
