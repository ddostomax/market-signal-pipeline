"""Thread-safe latency and throughput metrics."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Metrics:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _latencies_us: list[float] = field(default_factory=list, repr=False)
    processed: int = 0
    signals_emitted: int = 0
    warmup_skipped: int = 0
    _start: float = field(default_factory=time.perf_counter, repr=False)

    def record(self, latency_us: float, emitted: bool) -> None:
        with self._lock:
            self.processed += 1
            self._latencies_us.append(latency_us)
            if emitted:
                self.signals_emitted += 1
            else:
                self.warmup_skipped += 1

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            elapsed = max(time.perf_counter() - self._start, 1e-9)
            lat = np.asarray(self._latencies_us, dtype=np.float64)
            out = {
                "processed": float(self.processed),
                "signals_emitted": float(self.signals_emitted),
                "warmup_skipped": float(self.warmup_skipped),
                "throughput_msg_per_s": self.processed / elapsed,
                "elapsed_s": elapsed,
            }
            if lat.size:
                out.update(
                    {
                        "latency_p50_us": float(np.percentile(lat, 50)),
                        "latency_p99_us": float(np.percentile(lat, 99)),
                        "latency_max_us": float(lat.max()),
                    }
                )
            return out
