"""Entrypoint: train the signal model and persist it to disk.

Usage:
    python -m scripts.train_model            # default 20k ticks/symbol
    TICKS_PER_SECOND=... python -m scripts.train_model
"""
from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import argparse
import logging
import sys

from src.config import Config
from src.logging_config import configure_logging
from src.model.train import train

logger = logging.getLogger("train")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the market signal model")
    parser.add_argument("--ticks-per-symbol", type=int, default=20_000)
    args = parser.parse_args(argv)

    configure_logging()
    cfg = Config()
    result = train(cfg, ticks_per_symbol=args.ticks_per_symbol)

    print("\n=== Training complete ===")
    print(f"samples            : {result.n_samples:,}")
    print(f"baseline accuracy  : {result.baseline_accuracy:.4f}  (majority class)")
    print(f"model test accuracy: {result.accuracy:.4f}")
    print(f"lift over baseline : {result.lift:+.4f}")
    print(f"model saved        : {result.model_path}\n")
    print(result.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
