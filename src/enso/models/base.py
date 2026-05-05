"""Interface comum para modelos (baselines e DL)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ForecastModel(ABC):
    """Modelo de previsao escalar h passos a frente.

    Modelos podem operar sobre formato:
      - tabular: X shape [N, F]
      - sequencial: X shape [N, lookback, F]
    Especificado via atributo `input_kind` ('tabular' ou 'sequence').
    """
    name: str = "base"
    input_kind: str = "tabular"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None, **kwargs) -> "ForecastModel":
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...
