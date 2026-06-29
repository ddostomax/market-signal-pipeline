"""Streaming feature engine.

Maintains a fixed-size rolling window of recent prices/volumes per symbol and
computes a feature vector incrementally on each new tick. Designed so that the
exact same code is used both offline (training) and online (serving), which
eliminates train/serve skew.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "ret_1",
    "sma_ratio",
    "momentum",
    "volatility",
    "rsi",
    "volume_z",
)


@dataclass(slots=True)
class SymbolState:
    prices: deque[float]
    volumes: deque[float]


class FeatureEngine:
    """Per-symbol rolling feature computation.

    Not internally synchronized: each worker owns a disjoint set of symbols, so
    a given SymbolState is only ever touched by one thread (sharding gives us
    thread-safety without locks).
    """

    def __init__(self, short_window: int, long_window: int, rsi_window: int) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self._short = short_window
        self._long = long_window
        self._rsi = rsi_window
        self._capacity = max(long_window, rsi_window) + 1
        self._state: dict[str, SymbolState] = {}

    def _state_for(self, symbol: str) -> SymbolState:
        st = self._state.get(symbol)
        if st is None:
            st = SymbolState(
                prices=deque(maxlen=self._capacity),
                volumes=deque(maxlen=self._capacity),
            )
            self._state[symbol] = st
        return st

    def ready(self, symbol: str) -> bool:
        st = self._state.get(symbol)
        return st is not None and len(st.prices) >= self._capacity

    def update(self, symbol: str, price: float, volume: float) -> np.ndarray | None:
        """Append a new observation and return the feature vector, or None if
        the symbol is still warming up."""
        st = self._state_for(symbol)
        st.prices.append(price)
        st.volumes.append(volume)
        if len(st.prices) < self._capacity:
            return None
        return self._compute(st)

    def _compute(self, st: SymbolState) -> np.ndarray:
        prices = np.fromiter(st.prices, dtype=np.float64)
        volumes = np.fromiter(st.volumes, dtype=np.float64)

        returns = np.diff(prices) / prices[:-1]
        ret_1 = returns[-1]

        sma_short = prices[-self._short:].mean()
        sma_long = prices[-self._long:].mean()
        sma_ratio = (sma_short / sma_long) - 1.0

        momentum = (prices[-1] / prices[-self._short]) - 1.0
        volatility = returns[-self._long:].std()

        rsi = self._rsi_value(returns[-self._rsi:])

        vol_window = volumes[-self._long:]
        vol_std = vol_window.std()
        volume_z = 0.0 if vol_std == 0 else (volumes[-1] - vol_window.mean()) / vol_std

        return np.array(
            [ret_1, sma_ratio, momentum, volatility, rsi, volume_z],
            dtype=np.float64,
        )

    @staticmethod
    def _rsi_value(returns_window: np.ndarray) -> float:
        gains = returns_window[returns_window > 0].sum()
        losses = -returns_window[returns_window < 0].sum()
        if losses == 0:
            return 100.0
        rs = gains / losses
        return 100.0 - (100.0 / (1.0 + rs))
