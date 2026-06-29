# Interview Notes — Market Data Pipeline & ML Signal Engine

Everything you need to confidently talk about this project. Read top to bottom
once; memorize the **30-second pitch** and the **3 stories**.

---

## 1. 30-second elevator pitch

> "I built a real-time market-data pipeline that ingests price ticks, computes
> rolling technical features on the fly, and runs a machine-learning model that
> classifies the next short-term price move as up, down, or flat. The
> interesting part is the systems design: I shard the stream by symbol across a
> pool of worker threads — like Kafka partitions — so each symbol is processed
> in order by one thread while different symbols run in parallel, with no locks
> on the hot path. I instrumented it with p50/p99 latency and throughput, and
> it does roughly 1,800 messages/sec at ~1.4 ms median latency."

## 2. What I used (tech stack)

- **Python** — core language.
- **threading + queue.Queue** — producer/consumer concurrency, bounded queues
  for backpressure.
- **NumPy / Pandas** — vectorized feature math.
- **scikit-learn** — `HistGradientBoostingClassifier` inside a `Pipeline` with
  `StandardScaler`; `joblib` to persist the model.
- **Structured logging**, **dataclass-based config** (env-overridable),
  **pytest** for tests.

## 3. How it works (data flow, end to end)

1. **Generator** produces synthetic ticks per symbol using **geometric
   Brownian motion** plus a small **AR(1) momentum** term (so there's real,
   learnable signal — not a pure random walk).
2. **Feed (producer thread)** paces ticks at a target rate and routes each tick
   to a worker queue via `hash(symbol) % num_workers` (partitioning).
3. **Worker threads** each own a subset of symbols. For every tick they:
   update that symbol's **rolling FeatureEngine** → get a feature vector →
   call the **model** → emit a UP/DOWN/FLAT **signal** → record latency.
4. **SignalSink + Metrics** aggregate results thread-safely; at the end we print
   throughput and latency percentiles.
5. **Training** is offline: replay long series through the *same* FeatureEngine,
   label each point by its **forward return over a horizon** (UP/DOWN/FLAT),
   chronological 80/20 split, train, and compare to a majority-class baseline.

## 4. The three stories to tell (pick based on what they probe)

### Story A — Concurrency / systems (your strongest, most Google-relevant)
- **Problem:** features are stateful and order-dependent per symbol, so I can't
  just throw ticks at a shared thread pool — that would reorder a symbol's
  history and corrupt rolling features.
- **Solution:** **partition by symbol**. Hash each symbol to one worker. Now a
  symbol's ticks are strictly ordered (handled by one thread), while different
  symbols run concurrently. Each worker owns its own feature state, so the hot
  path is **lock-free** — no contention, no race conditions.
- **Why it matters:** this is exactly how Kafka consumer groups and sharded
  stream processors work; it scales horizontally by adding partitions/workers.
- **Backpressure:** bounded queues mean if a worker falls behind, the producer
  blocks instead of growing memory unboundedly.

### Story B — A real performance bug I found and fixed (great signal)
- **Symptom:** first run did only **78 msg/s** with **~39 ms** median latency.
- **Diagnosis:** scikit-learn's gradient-boosting `predict` uses **OpenMP**
  internally. With 4 worker threads each triggering OpenMP, I had **thread
  oversubscription** — way more native threads than cores, all fighting.
- **Fix:** pin `OMP_NUM_THREADS=1` (one math thread per Python worker). Result:
  **~1,800 msg/s, ~1.4 ms p50 — a ~23× improvement.**
- **Bonus depth:** Python's **GIL** means threads don't give true CPU
  parallelism for pure-Python work; for this workload the right production move
  is `multiprocessing` (or separate processes per partition, like real Kafka
  consumers), or exporting the model to **ONNX** for fast, GIL-free inference.

### Story C — Honest ML (shows maturity, avoids a classic trap)
- I **don't** claim to predict the real market. On a **pure random walk, no
  model can beat the majority class** — and my code proves it: set
  `momentum_phi=0` and accuracy collapses to baseline.
- To make the ML meaningful I injected a known **AR(1) momentum** structure;
  the model then beats the majority-class baseline by **~3.6 points**, and the
  per-class recall on UP/DOWN confirms it's learning the momentum, not cheating.
- I guarded against **look-ahead bias** with a chronological split and used the
  *same* feature code for train and serve to avoid **train/serve skew**.

## 5. Likely interview questions + crisp answers

**Q: Why threads and not async?**
A: The work per tick is CPU-bound (feature math + model inference), so asyncio
wouldn't help — async shines for I/O-bound waits. Threads + sharding give me
ordered, parallel-per-symbol processing; for more CPU scaling I'd go
multiprocessing.

**Q: How do you guarantee per-symbol ordering?**
A: Each symbol maps to exactly one worker via a hash, and that worker is
single-threaded over its queue, so a symbol's ticks are processed strictly in
arrival order.

**Q: What if one symbol is way hotter than others (skew)?**
A: Hash partitioning can hot-spot. I'd switch to key-based partitioning with
load stats, or finer-grained partitions than workers and rebalance — same as
Kafka partition assignment.

**Q: How do you avoid look-ahead bias?**
A: Labels use only *future* returns relative to each point, and I split
chronologically (train = earlier, test = later), never shuffling across time.

**Q: How would you scale this 100×?**
A: Replace in-process queues with Kafka partitions, run workers as separate
processes/pods (kills the GIL issue), batch inference or move to ONNX, add a
Redis feature cache, and export Prometheus metrics. The sharding model is
unchanged — that's the point.

**Q: How is it thread-safe without locks on the hot path?**
A: State is *partitioned*, not *shared*. The only shared structures are the
metrics and sink, which use a short lock; the per-tick feature/inference path
touches only data owned by one thread.

**Q: Why HistGradientBoosting?**
A: Strong tabular baseline, handles non-linear feature interactions, fast to
train. For latency-critical serving I'd benchmark a smaller model or ONNX.

## 6. Numbers to remember

- ~1,800 msg/s throughput, ~1.4 ms p50 / ~6 ms p99 latency (8 symbols, 4 workers).
- ~23× speedup from fixing thread oversubscription.
- Model ~0.39 accuracy vs ~0.35 majority baseline → +3.6 pts lift (3-class).

## 7. How it maps to the Google SWE-Intern JD

- *Concurrency / multi-threading / synchronization* → symbol-sharded worker pool,
  lock-free hot path, backpressure.
- *Distributed systems* → Kafka-style partitioning; clean path to multi-process / Kafka.
- *Performance, reliability, data analysis, debugging* → latency percentiles,
  the oversubscription bug hunt, metrics.
- *AI-integrated software* → online ML inference embedded in the data path.
- *Software design* → train/serve parity, validated config, structured logging, tests.

## 8. What NOT to say

- ❌ "It predicts the stock market / makes money." (It classifies synthetic data
  with injected structure.)
- ❌ Overstate the accuracy. Frame it as **beating the baseline on data with
  known structure**, and emphasize the engineering.
- ✅ Be ready to admit the GIL limitation and the synthetic-data caveat —
  volunteering limitations reads as senior, not weak.
