"""Synthetic market data generation.

Prices follow a geometric Brownian motion (GBM) per symbol, which gives
realistic-looking random walks with drift and volatility. This lets the whole
pipeline run deterministically and offline (no paid market-data feed, no
secrets) while still exercising every code path the way live data would.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Tick:
    symbol: str
    timestamp_ns: int
    price: float
    volume: int


class GBMTickGenerator:
    """Generates ticks for a universe of symbols using GBM dynamics."""

    def __init__(
        self,
        symbols: tuple[str, ...],
        seed: int = 42,
        annual_drift: float = 0.05,
        annual_vol: float = 0.35,
        steps_per_year: int = 252 * 6_000,
        momentum_phi: float = 0.35,
    ) -> None:
        """`momentum_phi` adds AR(1) autocorrelation to the log-return process:
        a fraction `phi` of the previous return carries into the next step. A
        pure random walk (phi=0) is unpredictable by construction; a small
        positive phi injects realistic, *learnable* short-term momentum so the
        ML model has genuine signal to extract."""
        if not symbols:
            raise ValueError("symbols must be non-empty")
        if not -1.0 < momentum_phi < 1.0:
            raise ValueError("momentum_phi must be in (-1, 1) for stationarity")
        self._symbols = symbols
        self._rng = np.random.default_rng(seed)
        # Start each symbol at a distinct, plausible price.
        self._prices = {
            sym: float(self._rng.uniform(80, 400)) for sym in symbols
        }
        self._last_return = {sym: 0.0 for sym in symbols}
        self._phi = momentum_phi
        dt = 1.0 / steps_per_year
        self._mu_dt = (annual_drift - 0.5 * annual_vol**2) * dt
        self._sigma_sqrt_dt = annual_vol * np.sqrt(dt)

    def next_tick(self, symbol: str) -> Tick:
        """Advance one symbol by a single (AR(1)-augmented) GBM step."""
        z = self._rng.standard_normal()
        innovation = self._mu_dt + self._sigma_sqrt_dt * z
        log_return = self._phi * self._last_return[symbol] + innovation
        self._last_return[symbol] = log_return
        new_price = self._prices[symbol] * float(np.exp(log_return))
        self._prices[symbol] = new_price
        volume = int(self._rng.integers(100, 5_000))
        return Tick(
            symbol=symbol,
            timestamp_ns=time.time_ns(),
            price=round(new_price, 4),
            volume=volume,
        )

    def stream(self, total_ticks: int) -> list[Tick]:
        """Produce a fixed batch of ticks across all symbols (round-robin)."""
        out: list[Tick] = []
        n_sym = len(self._symbols)
        for i in range(total_ticks):
            sym = self._symbols[i % n_sym]
            out.append(self.next_tick(sym))
        return out
