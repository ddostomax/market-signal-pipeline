"""Market data feed: a producer thread that dispatches ticks to per-worker
queues, partitioned by symbol (the same idea as Kafka topic partitions).

Partitioning by symbol guarantees that all ticks for a given symbol are
processed in order by exactly one worker, while different symbols are
processed concurrently across workers.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Optional

from .generator import GBMTickGenerator, Tick

logger = logging.getLogger(__name__)

# Sentinel pushed to a worker queue to signal graceful shutdown.
SHUTDOWN = None


def partition_for(symbol: str, num_partitions: int) -> int:
    """Deterministically map a symbol to a partition/worker index."""
    return (hash(symbol) & 0x7FFFFFFF) % num_partitions


class MarketDataFeed:
    """Generates ticks at a target rate and routes them to worker queues."""

    def __init__(
        self,
        generator: GBMTickGenerator,
        symbols: tuple[str, ...],
        worker_queues: list["queue.Queue[Optional[Tick]]"],
        ticks_per_second: int,
    ) -> None:
        self._generator = generator
        self._symbols = symbols
        self._queues = worker_queues
        self._ticks_per_second = ticks_per_second
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.produced = 0

    def _run(self, total_ticks: int) -> None:
        interval = 1.0 / self._ticks_per_second
        next_emit = time.perf_counter()
        n_sym = len(self._symbols)
        for i in range(total_ticks):
            if self._stop.is_set():
                break
            sym = self._symbols[i % n_sym]
            tick = self._generator.next_tick(sym)
            part = partition_for(sym, len(self._queues))
            # Backpressure: block briefly if a worker is saturated.
            self._queues[part].put(tick)
            self.produced += 1

            # Simple rate limiting / pacing.
            next_emit += interval
            sleep_for = next_emit - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
        # Signal every worker to drain and exit.
        for q in self._queues:
            q.put(SHUTDOWN)
        logger.info("feed_complete", extra={"produced": self.produced})

    def start(self, total_ticks: int) -> None:
        self._thread = threading.Thread(
            target=self._run, args=(total_ticks,), name="market-feed", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)
