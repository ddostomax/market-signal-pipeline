"""Online inference wrapper around the trained model.

Loaded once and shared (read-only) across worker threads. scikit-learn's
predict path is thread-safe for concurrent reads, so no locking is required.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import joblib
import numpy as np

from .labeling import CLASS_NAMES

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    label: int
    label_name: str
    confidence: float


class SignalPredictor:
    def __init__(self, model, feature_names: tuple[str, ...]) -> None:
        self._model = model
        self._feature_names = feature_names

    @classmethod
    def load(cls, path: str) -> "SignalPredictor":
        try:
            blob = joblib.load(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Model not found at {path!r}. Run scripts/train_model.py first."
            ) from exc
        return cls(blob["model"], blob["features"])

    def predict(self, symbol: str, features: np.ndarray) -> Signal:
        proba = self._model.predict_proba(features.reshape(1, -1))[0]
        label = int(proba.argmax())
        return Signal(
            symbol=symbol,
            label=label,
            label_name=CLASS_NAMES[label],
            confidence=float(proba[label]),
        )
