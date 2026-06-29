# Real-Time Market Data Pipeline & ML Signal Engine

A multithreaded, **partition-sharded** streaming pipeline that ingests market
ticks, computes rolling features online, and serves a machine-learning model
that classifies short-term price movement (**UP / DOWN / FLAT**) — with built-in
latency and throughput instrumentation.

The project is intentionally **self-contained and deterministic**: it generates
its own synthetic market data, so it runs offline with no API keys, no secrets,
and no paid data feed, while still exercising the exact code paths a live feed
would.

---

## Why this design

The headline idea is to treat this like a real low-latency trading-infra
component, not a notebook:

- **Sharding by symbol (Kafka-style partitioning).** Each symbol is hashed to
  exactly one worker, so all ticks for a symbol are processed *in order* by one
  thread, while different symbols run *concurrently*. This gives a **lock-free
  hot path** — no two threads ever touch the same symbol's state.
- **Train/serve parity.** The *same* `FeatureEngine` is used during offline
  training and online serving, eliminating train/serve skew (a classic ML bug).
- **Honest evaluation.** Chronological train/test split (no shuffling) to avoid
  look-ahead bias, and every result is compared against a **majority-class
  baseline**.

```
                       ┌────────────────────────────────────────────┐
                       │              MarketDataFeed                  │
   GBM generator ─────►│  (producer thread, paces ticks, routes by   │
                       │   hash(symbol) % N  → per-worker queues)     │
                       └───────┬───────────┬───────────┬─────────────┘
                               │           │           │   (bounded queues
                               ▼           ▼           ▼    = backpressure)
                          ┌────────┐  ┌────────┐  ┌────────┐
                          │Worker 0│  │Worker 1│  │Worker N│   each owns:
                          │ feats  │  │ feats  │  │ feats  │   - FeatureEngine
                          │ +infer │  │ +infer │  │ +infer │   - subset of symbols
                          └───┬────┘  └───┬────┘  └───┬────┘
                              └───────────┼───────────┘
                                          ▼
                            thread-safe SignalSink + Metrics
                              (p50/p99 latency, throughput)
```

## Project layout

```
src/
  config.py              # env-overridable, validated configuration
  logging_config.py      # structured key=value logging
  metrics.py             # thread-safe latency/throughput tracking
  data/
    generator.py         # GBM + AR(1) momentum synthetic tick generator
    feed.py              # producer thread + symbol partitioning
  features/
    engine.py            # incremental rolling-window feature computation
  model/
    labeling.py          # forward-return -> UP/DOWN/FLAT labels
    train.py             # offline training (chronological split, baseline)
    predictor.py         # thread-safe online inference wrapper
  pipeline/
    worker.py            # consumer worker thread
    orchestrator.py      # wires feed + workers + sink + metrics
scripts/
  train_model.py         # train and persist the model
  run_pipeline.py        # run the live (replayed) pipeline
tests/                   # feature, labeling, partitioning, pipeline tests
```

## Features computed (per tick, online)

`ret_1`, `sma_ratio` (short/long SMA), `momentum`, `volatility` (rolling std of
returns), `rsi` (Wilder-style), `volume_z` (volume z-score).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Train the model (writes artifacts/model.joblib)
python -m scripts.train_model --ticks-per-symbol 20000

# 2) Run the streaming pipeline
python -m scripts.run_pipeline --ticks 40000

# 3) Tests
pip install pytest && pytest -q
```

Everything is configurable via environment variables (see `src/config.py`),
e.g. `NUM_WORKERS=8 TICKS_PER_SECOND=2000000 python -m scripts.run_pipeline`.

## Representative results (8 symbols, 4 workers, laptop)

| Metric                          | Value          |
|---------------------------------|----------------|
| Model test accuracy             | ~0.39          |
| Majority-class baseline         | ~0.35          |
| **Lift over baseline**          | **+3.6 pts**   |
| Throughput                      | ~1,800 msg/s   |
| Latency p50 / p99               | ~1.4 / ~6.2 ms |

> A ~23× throughput gain came from **pinning native thread pools**
> (`OMP_NUM_THREADS=1`): without it, each of the 4 worker threads spawned its
> own OpenMP pool, oversubscribing cores and inflating p50 latency to ~39 ms.

## Honest notes / limitations

- The data is **synthetic** with a deliberately injected AR(1) momentum
  component, so the model has *learnable* structure. On a **pure random walk no
  model can beat the majority class** — and the code demonstrates exactly that
  when `momentum_phi=0`. This is the point: the engineering is the deliverable,
  **not** a claim of real-world trading profit.
- Python's GIL limits CPU parallelism across threads; the design scales out
  cleanly to `multiprocessing` or real Kafka partitions (the sharding model is
  identical). See `INTERVIEW_NOTES.md`.

## Possible extensions

Real feed adapter (websocket), ONNX/batched inference, Redis feature cache,
Prometheus metrics export, multiprocessing or Kafka-backed partitions.
