import numpy as np

from src.config import Config
from src.data.feed import partition_for
from src.features.engine import FEATURE_NAMES
from src.model.predictor import SignalPredictor
from src.pipeline.orchestrator import Pipeline


class _StubModel:
    """Deterministic stand-in so the pipeline test needs no trained artifact."""

    def predict_proba(self, X):
        n = X.shape[0]
        return np.tile(np.array([0.2, 0.3, 0.5]), (n, 1))


def test_partitioning_is_deterministic_and_in_range():
    for sym in ("AAPL", "MSFT", "GOOG"):
        p1 = partition_for(sym, 4)
        p2 = partition_for(sym, 4)
        assert p1 == p2
        assert 0 <= p1 < 4


def test_pipeline_processes_all_ticks():
    cfg = Config(ticks_per_second=1_000_000, num_workers=3)
    predictor = SignalPredictor(_StubModel(), FEATURE_NAMES)
    pipeline = Pipeline(cfg, predictor)
    total = 5_000
    snap = pipeline.run(total_ticks=total)
    assert int(snap["processed"]) == total
    assert snap["signals_emitted"] + snap["warmup_skipped"] == total
