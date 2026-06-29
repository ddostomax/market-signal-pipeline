"""Central, validated configuration for the pipeline.

Values can be overridden via environment variables so the same code runs
unchanged in local, container, and CI environments.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {raw!r}") from exc


@dataclass(frozen=True)
class Config:
    # Universe / feed
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "META", "TSLA", "NFLX")
    ticks_per_second: int = field(default_factory=lambda: _env_int("TICKS_PER_SECOND", 5000))
    seed: int = field(default_factory=lambda: _env_int("SEED", 42))

    # Streaming feature windows (in ticks)
    short_window: int = field(default_factory=lambda: _env_int("SHORT_WINDOW", 5))
    long_window: int = field(default_factory=lambda: _env_int("LONG_WINDOW", 30))
    rsi_window: int = field(default_factory=lambda: _env_int("RSI_WINDOW", 14))

    # Labeling for supervised training
    horizon: int = field(default_factory=lambda: _env_int("HORIZON", 10))
    move_threshold_bps: float = field(default_factory=lambda: _env_float("MOVE_THRESHOLD_BPS", 5.0))

    # Concurrency
    num_workers: int = field(default_factory=lambda: _env_int("NUM_WORKERS", 4))
    queue_maxsize: int = field(default_factory=lambda: _env_int("QUEUE_MAXSIZE", 10000))

    # Model
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "artifacts/model.joblib"))

    def __post_init__(self) -> None:
        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window")
        if self.num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if self.ticks_per_second < 1:
            raise ValueError("ticks_per_second must be >= 1")
        if not self.symbols:
            raise ValueError("at least one symbol is required")

    @property
    def warmup_ticks(self) -> int:
        """Minimum history per symbol before features are valid."""
        return max(self.long_window, self.rsi_window) + 1


DEFAULT_CONFIG = Config()
