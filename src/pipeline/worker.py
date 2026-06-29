"""Consumer worker.

Each worker owns one input queue (one partition). It maintains the feature
state for the symbols routed to it, runs inference, and pushes resulting
signals to a shared, thread-safe sink. Because symbols are sharded across
workers, no two threads ever touch the same symbol state -> lock-free hot path.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

from ..data.feed import SHUTDOWN
from ..data.generator import Tick
from ..features.engine import FeatureEngine
from ..metrics import Metrics
from ..model.predictor import Signal, SignalPredictor

logger = logging.getLogger(__name__)


class Worker(threading.Thread):
    def __init__(
        self,
        worker_id: int,
        in_queue: "queue.Queue[Optional[Tick]]",
        predictor: SignalPredictor,
        feature_engine: FeatureEngine,
        metrics: Metrics,
        on_signal: Callable[[Signal], None],
    ) -> None:
        super().__init__(name=f"worker-{worker_id}", daemon=True)
        self._id = worker_id
        self._in = in_queue
        self._predictor = predictor
        self._features = feature_engine
        self._metrics = metrics
        self._on_signal = on_signal

    def run(self) -> None:
        while True:
            tick = self._in.get()
            if tick is SHUTDOWN:
                self._in.task_done()
                break
            start = time.perf_counter()
            try:
                fv = self._features.update(tick.symbol, tick.price, float(tick.volume))
                emitted = False
                if fv is not None:
                    signal = self._predictor.predict(tick.symbol, fv)
                    self._on_signal(signal)
                    emitted = True
            except Exception:  # never let one bad tick kill the worker
                logger.exception("worker_error", extra={"worker": self._id})
                self._in.task_done()
                continue
            latency_us = (time.perf_counter() - start) * 1e6
            self._metrics.record(latency_us, emitted)
            self._in.task_done()
        logger.info("worker_stopped", extra={"worker": self._id})
