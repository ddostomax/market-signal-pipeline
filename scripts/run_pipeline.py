"""Entrypoint: run the live (replayed) pipeline and print a metrics report.

Usage:
    python -m scripts.run_pipeline --ticks 50000
"""
from __future__ import annotations

import os

# Pin native thread pools BEFORE importing numpy/sklearn. Each worker thread
# runs its own single-row inference, so letting BLAS/OpenMP each spawn a pool
# causes thread oversubscription and destroys latency. One math thread per
# Python worker is the right model here.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import argparse
import logging
import sys

from src.config import Config
from src.logging_config import configure_logging
from src.model.predictor import SignalPredictor
from src.pipeline.orchestrator import Pipeline

logger = logging.getLogger("run")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the market signal pipeline")
    parser.add_argument("--ticks", type=int, default=50_000, help="total ticks to replay")
    args = parser.parse_args(argv)

    configure_logging()
    cfg = Config()

    try:
        predictor = SignalPredictor.load(cfg.model_path)
    except FileNotFoundError as exc:
        logger.error("model_missing", extra={"error": str(exc)})
        return 1

    pipeline = Pipeline(cfg, predictor)
    snap = pipeline.run(total_ticks=args.ticks)

    print("\n=== Pipeline report ===")
    print(f"workers            : {cfg.num_workers}")
    print(f"symbols            : {len(cfg.symbols)}")
    print(f"processed          : {int(snap['processed']):,}")
    print(f"signals emitted    : {int(snap['signals_emitted']):,}")
    print(f"warmup skipped     : {int(snap['warmup_skipped']):,}")
    print(f"throughput (msg/s) : {snap['throughput_msg_per_s']:,.0f}")
    if "latency_p50_us" in snap:
        print(f"latency p50 (us)   : {snap['latency_p50_us']:.1f}")
        print(f"latency p99 (us)   : {snap['latency_p99_us']:.1f}")
        print(f"latency max (us)   : {snap['latency_max_us']:.1f}")
    print(f"signal distribution: {snap['signal_distribution']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
