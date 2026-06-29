"""Offline training.

Builds a labeled dataset by replaying GBM price series through the *same*
streaming FeatureEngine used at serving time, then trains a gradient-boosted
classifier. Uses a chronological train/test split (no shuffling) to avoid
look-ahead bias.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Config
from ..data.generator import GBMTickGenerator
from ..features.engine import FEATURE_NAMES, FeatureEngine
from .labeling import CLASS_NAMES, forward_return_labels

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainResult:
    accuracy: float
    baseline_accuracy: float
    n_samples: int
    report: str
    model_path: str

    @property
    def lift(self) -> float:
        """Absolute accuracy gain over the majority-class baseline."""
        return self.accuracy - self.baseline_accuracy


def _build_dataset(cfg: Config, ticks_per_symbol: int) -> tuple[np.ndarray, np.ndarray]:
    """Replay long series per symbol and emit (features, label) pairs."""
    gen = GBMTickGenerator(cfg.symbols, seed=cfg.seed)
    feats: list[np.ndarray] = []
    labels: list[int] = []

    for symbol in cfg.symbols:
        engine = FeatureEngine(cfg.short_window, cfg.long_window, cfg.rsi_window)
        prices: list[float] = []
        per_tick_features: list[np.ndarray | None] = []
        for _ in range(ticks_per_symbol):
            tick = gen.next_tick(symbol)
            prices.append(tick.price)
            per_tick_features.append(engine.update(symbol, tick.price, tick.volume))

        price_arr = np.asarray(prices, dtype=np.float64)
        label_arr = forward_return_labels(price_arr, cfg.horizon, cfg.move_threshold_bps)
        for i, fv in enumerate(per_tick_features):
            if fv is not None and label_arr[i] != -1:
                feats.append(fv)
                labels.append(int(label_arr[i]))

    X = np.vstack(feats)
    y = np.asarray(labels, dtype=np.int64)
    return X, y


def train(cfg: Config, ticks_per_symbol: int = 20_000) -> TrainResult:
    logger.info("building_dataset", extra={"symbols": len(cfg.symbols)})
    X, y = _build_dataset(cfg, ticks_per_symbol)

    # Chronological split per the original ordering (no shuffle).
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.08,
                    max_depth=6,
                    l2_regularization=1.0,
                    random_state=cfg.seed,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # Majority-class baseline on the test split (honest comparison point).
    values, counts = np.unique(y_train, return_counts=True)
    majority_class = values[counts.argmax()]
    baseline_acc = accuracy_score(y_test, np.full_like(y_test, majority_class))
    report = classification_report(
        y_test,
        preds,
        labels=sorted(CLASS_NAMES),
        target_names=[CLASS_NAMES[c] for c in sorted(CLASS_NAMES)],
        zero_division=0,
    )

    os.makedirs(os.path.dirname(cfg.model_path) or ".", exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_NAMES}, cfg.model_path)
    logger.info("model_saved", extra={"path": cfg.model_path, "accuracy": acc})

    return TrainResult(
        accuracy=float(acc),
        baseline_accuracy=float(baseline_acc),
        n_samples=int(len(X)),
        report=report,
        model_path=cfg.model_path,
    )
