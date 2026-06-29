"""Wires the feed, worker pool, model, and metrics into a running pipeline."""
from __future__ import annotations

import logging
import queue
import threading
from collections import Counter
from typing import Optional

from ..config import Config
from ..data.feed import MarketDataFeed
from ..data.generator import GBMTickGenerator, Tick
from ..features.engine import FeatureEngine
from ..metrics import Metrics
from ..model.predictor import Signal, SignalPredictor
from .worker import Worker

logger = logging.getLogger(__name__)


class SignalSink:
    """Thread-safe collector for emitted signals (stand-in for a downstream
    bus / order router). Keeps only aggregate counts to stay memory-bounded."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counts: Counter[str] = Counter()
        self.last: dict[str, Signal] = {}

    def __call__(self, signal: Signal) -> None:
        with self._lock:
            self.counts[signal.label_name] += 1
            self.last[signal.symbol] = signal


class Pipeline:
    def __init__(self, cfg: Config, predictor: SignalPredictor) -> None:
        self._cfg = cfg
        self._predictor = predictor
        self._metrics = Metrics()
        self._sink = SignalSink()
        self._queues: list["queue.Queue[Optional[Tick]]"] = [
            queue.Queue(maxsize=cfg.queue_maxsize) for _ in range(cfg.num_workers)
        ]
        self._workers: list[Worker] = []

    @property
    def metrics(self) -> Metrics:
        return self._metrics

    @property
    def sink(self) -> SignalSink:
        return self._sink

    def run(self, total_ticks: int) -> dict:
        cfg = self._cfg
        for wid in range(cfg.num_workers):
            engine = FeatureEngine(cfg.short_window, cfg.long_window, cfg.rsi_window)
            worker = Worker(
                worker_id=wid,
                in_queue=self._queues[wid],
                predictor=self._predictor,
                feature_engine=engine,
                metrics=self._metrics,
                on_signal=self._sink,
            )
            worker.start()
            self._workers.append(worker)

        generator = GBMTickGenerator(cfg.symbols, seed=cfg.seed)
        feed = MarketDataFeed(generator, cfg.symbols, self._queues, cfg.ticks_per_second)
        feed.start(total_ticks)
        feed.join()

        for worker in self._workers:
            worker.join()

        snap = self._metrics.snapshot()
        snap["signal_distribution"] = dict(self._sink.counts)
        logger.info("pipeline_complete", extra={"snapshot": snap})
        return snap
