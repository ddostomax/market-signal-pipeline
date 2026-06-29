import numpy as np

from src.features.engine import FEATURE_NAMES, FeatureEngine
from src.model.labeling import DOWN, UP, forward_return_labels


def test_warmup_returns_none_until_capacity():
    eng = FeatureEngine(short_window=3, long_window=10, rsi_window=8)
    out = None
    for i in range(10):
        out = eng.update("AAPL", price=100.0 + i, volume=1000)
    # capacity = max(long, rsi)+1 = 11, so after 10 ticks still warming up
    assert out is None
    assert not eng.ready("AAPL")


def test_feature_vector_shape_and_finiteness():
    eng = FeatureEngine(short_window=3, long_window=10, rsi_window=8)
    fv = None
    rng = np.random.default_rng(0)
    price = 100.0
    for _ in range(50):
        price *= 1 + rng.normal(0, 0.01)
        fv = eng.update("AAPL", price=price, volume=int(rng.integers(100, 1000)))
    assert fv is not None
    assert fv.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(fv))


def test_labels_uptrend_and_downtrend():
    up = np.array([100.0 * (1.01**i) for i in range(50)])
    labels = forward_return_labels(up, horizon=5, threshold_bps=5.0)
    assert labels[0] == UP

    down = np.array([100.0 * (0.99**i) for i in range(50)])
    labels = forward_return_labels(down, horizon=5, threshold_bps=5.0)
    assert labels[0] == DOWN
