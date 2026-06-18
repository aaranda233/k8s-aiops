# K8s-AIOps: Event Collection, Log Parsing, and Unsupervised Anomaly Detection

**Companion to:** [RESEARCH.md](./RESEARCH.md) — covers Layers 1 and 2 of the pipeline.
**Source files:** `src/collector/`, `src/parser/`, `src/detector/`, `src/pipeline.py`

---

## Abstract

This document describes the first two layers of the K8s-AIOps pipeline: event collection from the Kubernetes API and unsupervised anomaly detection using Isolation Forest. The system ingests a continuous stream of Kubernetes events, abstracts them into log templates via online log parsing (Drain3), aggregates templates into fixed-width time windows, and scores each window **per namespace** with a three-signal ensemble — a periodically retrained Isolation Forest (absolute `decision_function` score), a log-severity signal, and a template-novelty signal. The window score is the maximum over its namespaces and the deviating namespace is recorded as the culprit. Windows whose anomaly score exceeds a configurable threshold are routed to the fine-tuned SLM described in RESEARCH.md. The entire detection path runs without labeled data.

---

## 1. Layer 1 — Event Collection

### 1.1 Kubernetes Watch API vs. Polling

Kubernetes exposes cluster state changes through its Watch API: a long-lived HTTP connection using chunked transfer encoding that pushes events as JSON objects in real time. This avoids polling latency and reduces API server load:

```
GET /api/v1/events?watch=true&resourceVersion=<rv>

← {"type":"ADDED","object":{...,"reason":"OOMKilling","message":"..."}}
← {"type":"MODIFIED","object":{...,"reason":"BackOff","message":"..."}}
← {"type":"ADDED","object":{...,"reason":"Pulled","message":"..."}}
  [stream continues]
```

The `resourceVersion` field is a monotonic cursor. If the connection drops, the collector reconnects from the last seen `resourceVersion`, guaranteeing at-least-once delivery of all events. When `resourceVersion` has expired from the API server's cache (HTTP 410), the collector resets to the current version, accepting a potential gap.

### 1.2 LogEntry Schema

Each Kubernetes event is normalized into a `LogEntry` dataclass (`src/collector/k8s_collector.py`):

```python
@dataclass
class LogEntry:
    timestamp: float    # epoch seconds (from event.last_timestamp)
    namespace: str      # e.g. "production"
    source: str         # "Pod/api-gateway-7d9f8c-xk2p"
    reason: str         # "OOMKilling", "BackOff", "Pulled"...
    message: str        # raw event message
    raw: str            # "{namespace} {source} {reason} {message}"
    event_type: str     # "ADDED" | "MODIFIED" | "DELETED"
```

Only `ADDED` and `MODIFIED` events are forwarded downstream. `DELETED` events carry no diagnostic information.

### 1.3 Masking in the Raw Field

The `raw` field is a single string that concatenates namespace, source, reason and message. Its purpose is to feed Drain3 (log parser) rather than to be human-readable. Variable tokens (IPs, UUIDs, pod suffixes, numbers, versions) are masked by the parser layer, not at collection time, so the raw field always contains full information.

### 1.4 Namespace Scoping

The collector supports two scoping modes:

| Mode | Behaviour |
|------|-----------|
| `namespaces=None` | `list_event_for_all_namespaces` — single global Watch stream |
| `namespaces=[...]` | Per-namespace Watch with `timeout_seconds=5` round-robin rotation |

Global mode is preferred in production clusters. Namespace filtering reduces noise in multi-tenant environments at the cost of N concurrent Watch connections.

### 1.5 Dual Execution Modes

| Mode | API call | Use case |
|------|----------|----------|
| `fetch_events_snapshot()` | `list_namespaced_event(limit=500)` | Bootstrap, replay, testing |
| `stream_events()` | `watch.Watch().stream(...)` | Production live monitoring |

In live mode (`run_live`), the pipeline first calls `fetch_events_snapshot()` to pre-load historical events into the bootstrap buffer, then switches to `stream_events()`. This ensures the Isolation Forest is fully trained on real baseline data before the Watch stream begins.

---

## 2. Layer 1b — Online Log Parsing with Drain3

### 2.1 Motivation

Kubernetes event messages are semi-structured: the `reason` field is categorical (`OOMKilling`, `BackOff`, `Pulled`) but the `message` field is free text containing variable tokens:

```
Back-off restarting failed container
Failed to pull image "api-gateway:v2.1.3-rc4": rpc error: code = NotFound
Memory cgroup out of memory: Kill process 18234 (node) score 987
```

Passing raw messages directly to Isolation Forest would create an unbounded, sparse feature space. Log parsing reduces this to a compact set of recurring templates.

### 2.2 Drain3 Algorithm

Drain3 is an online log parsing algorithm that maintains a prefix tree (trie) of log templates. Each leaf represents a log cluster with a representative template and a set of matching log entries. When a new log arrives:

1. The log is tokenized by whitespace.
2. The prefix tree is traversed using the first tokens as keys.
3. A similarity score is computed between the log and candidate clusters at the matching leaf: `sim = (matching tokens) / (total tokens)`.
4. If `sim ≥ sim_th` (threshold), the log is assigned to the existing cluster (template updated if needed). Otherwise, a new cluster is created.

This runs in O(depth × candidates) per log, making it suitable for real-time ingestion.

### 2.3 Configuration

```python
cfg.drain_sim_th = 0.4        # minimum similarity to join existing cluster
cfg.drain_depth  = 4          # prefix tree depth (token positions used as keys)
cfg.drain_max_children = 100  # max branches per tree node
```

Masking rules applied before template matching:

| Pattern | Mask |
|---------|------|
| `(\d{1,3}\.){3}\d{1,3}(:\d+)?` | `IP` |
| `[0-9a-f]{8,}-[0-9a-f\-]{8,}` | `UUID` |
| `(-[a-z0-9]{5,10}){2,}` | `POD_SUFFIX` |
| `\b\d+\b` | `NUM` |
| `:\d+\.\d+[\.\d]*` | `VER` |

Example:

```
Input:  "Failed to pull image api-gateway:v2.1.3: rpc error: code = NotFound"
Masked: "Failed to pull image api-gateway<VER> rpc error: code = NotFound"
Output: cluster_id=7, template="Failed to pull image <*> rpc error: code = <*>"
```

### 2.4 ParsedLog Output

```python
@dataclass(frozen=True)
class ParsedLog:
    cluster_id: int     # integer ID of the matched Drain3 cluster
    template: str       # abstracted template string
    raw: str            # original raw string (preserved for RCA prompt)
    namespace: str
    timestamp: float
```

The `cluster_id` is the key feature used by the Isolation Forest. The number of distinct `cluster_id` values grows over time as new log patterns are encountered — the feature space is dynamic.

---

## 3. Layer 2a — Time Window Aggregation

### 3.1 Fixed-Width Sliding Windows

Parsed logs are aggregated into non-overlapping time windows of fixed duration (default: 60 seconds). Each window captures:

```python
@dataclass
class WindowData:
    index: int                        # sequential window number
    start_time: float                 # epoch seconds
    end_time: float
    raw_logs: list[str]               # for RCA prompt construction
    namespaces: set[str]              # namespaces active in this window
    cluster_counts: dict[int, int]    # {cluster_id: count} — the feature vector
    anomaly_score: float
    is_anomaly: bool
```

`cluster_counts` is a sparse frequency map: how many times each Drain3 template appeared in this 60-second window. This becomes the feature vector for Isolation Forest.

### 3.2 WindowBuilder

`WindowBuilder` assigns each `ParsedLog` to the appropriate window based on its timestamp:

```python
window_idx = int((timestamp - t_start) / window_size)
```

If a log's `window_idx` exceeds the current open window's index, the current window is closed (returned to the pipeline) and a new one is opened. This enables streaming operation without buffering future events.

A periodic flush timer (`_start_window_flush_timer`) closes the current window every `window_size_seconds` even if no new events arrive, preventing indefinitely open windows during quiet periods.

### 3.3 Feature Representation

For a cluster with N distinct Drain3 templates seen across all history, a window W is represented as a sparse vector **f** ∈ ℝᴺ where:

```
f[j] = count of cluster_id j in window W
```

Before training and scoring, vectors are L1-normalized so that windows with different event volumes are comparable:

```
f_normalized[j] = f[j] / Σ f[k]
```

This normalization is critical: a busy window with 500 events and a quiet window with 20 events should be compared by their _distribution_ of templates, not by raw counts.

The window also keeps the per-template counts **broken down by namespace** (`ns_cluster_counts`). The Isolation Forest is trained and scored on these **per-namespace** vectors (one row per `(window, namespace)`), so detection localizes to the service that actually deviates rather than to the cluster as a whole (§4.1).

---

## 4. Layer 2b — Isolation Forest Anomaly Detection

### 4.1 Algorithm Overview

Isolation Forest (Liu et al., 2008) is an ensemble anomaly detection method based on the observation that anomalies are few and different — they require fewer random splits to be isolated than normal points. The algorithm:

1. Builds an ensemble of `n_estimators` random isolation trees.
2. Each tree recursively partitions the feature space by selecting a random feature and a random split value within the observed range.
3. The isolation depth of a point is the average number of splits required to isolate it across all trees.
4. Shallow isolation depth → anomaly (easy to isolate). Deep → normal (hard to isolate).

**Absolute scoring (revised).** An earlier version normalized the raw `score_samples()` output **min-max within the current history buffer**. That made the score *relative*: the least-normal window of the moment was always mapped to ≈1.0, even when every window was perfectly normal — the dominant cause of an "anomaly flood" on a busy cluster. The detector now derives an **absolute** anomaly score from scikit-learn's `decision_function` (which is calibrated by `contamination`, so `d ≥ 0` is inside the normal manifold and `d < 0` is anomalous):

```
d = model.decision_function(x)          # ≥ 0 normal, < 0 anomalous
if_score = 0           if d ≥ 0
         = clip(-d / d_scale, 0, 1)    otherwise
```

where `d_scale = max(-min(decision_function(X_train)), 0.05)` is fixed at training time (the depth of the worst normal training point). A normal window now scores low and *varied* instead of saturating to 1.0.

**Three-signal ensemble, scored per namespace.** Isolation Forest alone only sees the *template distribution*. Two complementary signals cover its blind spots, and **all three are computed per namespace** so a healthy namespace cannot drag the whole window into anomaly:

| Signal | Detects | Definition (per namespace) |
|--------|---------|----------------------------|
| **Isolation Forest** | unusual template distribution | `if_score` above, on the namespace's per-template vector |
| **Severity** | a service whose logs are mostly errors | mapped error ratio `err/total` (FATAL/ERROR/CRITICAL), thresholds 0.25→0.70 |
| **Novelty** | a burst of never-before-seen templates | ratio of logs whose `cluster_id ∉ trained_ids`, thresholds 0.15→0.50 |

The unit of analysis is therefore `(namespace, window)`. Each namespace `ns` in the window gets `s(ns) = max(if, severity, novelty·warmup)`; the **window score is `max` over namespaces** and the namespace achieving it is recorded as the **culprit** (`ScoredWindow.culprit_namespace`), which becomes the focus of the downstream RCA. A single-namespace window reduces exactly to the previous whole-window behaviour. `warmup` is a ramp (default 20 windows) that damps *only* novelty right after a cold start — when the parser and model are empty, everything is "new" and novelty is uninformative.

### 4.2 Configuration

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `n_estimators` | 200 | More trees → more stable scores. 200 is sufficient for feature spaces of ~50-200 templates |
| `contamination` | 0.05 | Assumes 5% of training rows are anomalous. Calibrates `decision_function`, which the **absolute** `if_score` is derived from |
| `threshold` | 0.80 | Anomaly score above which a (namespace, window) triggers an alert |
| `bootstrap_windows` | 10 | Minimum windows before first model fit |
| `rolling_window_size` | 50 | Size of the sliding history buffer for retraining |
| `retrain_every_n` | 5 | Retrain after every 5 new windows |
| `novelty_warmup_windows` | 20 | Windows after (re)start during which the novelty signal is ramped 0→1 (cold start = everything is "new") |

### 4.3 Three-Phase Operation

#### Phase 1 — Bootstrap

The first `bootstrap_windows` (default: 10) windows are assumed to represent normal cluster behavior. They are accumulated without scoring and used to fit the initial Isolation Forest model. During bootstrap, the pipeline emits progress events to the UI but does not raise alerts.

In live mode, the historical snapshot pre-loaded at startup typically covers this bootstrap phase without delay.

#### Phase 2 — Detection

Each new window W is scored **per namespace**, using the **current** model's feature set (the Drain3 cluster IDs seen at last training time). New cluster IDs discovered after the last retraining are silently ignored by the IF until the next retrain — but caught immediately by the novelty signal:

```python
for ns in W.namespaces:
    vec      = vectorize(W.ns_cluster_counts[ns], feature_ids=trained_cluster_ids)
    if_score = if_anom(model.decision_function(vec))      # absolute, vs training
    sev      = severity(W, ns)                            # local error ratio
    nov      = novelty(W, ns) * warmup_factor()           # unseen-template ratio
    s[ns]    = max(if_score, sev, nov)
score_W  = max(s.values())          # window score
culprit  = argmax(s)                # namespace to focus the RCA on
```

The score is **absolute** (referenced to the training distribution, not the current buffer), so a quiet, well-behaved cluster produces low scores and only a genuinely anomalous namespace crosses the `0.80` threshold.

#### Phase 3 — Periodic Retraining

Every `retrain_every_n` windows, the model is retrained on the last `rolling_window_size` windows from the history buffer:

```python
if since_last_retrain >= retrain_every_n:
    train(list(history_deque))   # maxlen=rolling_window_size
    model_version += 1
```

This sliding window approach allows the detector to adapt to gradual changes in cluster behavior (deployment additions, traffic growth) while discarding stale patterns. The model version counter is included in every scored window and RCA event for traceability.

**Risk:** If a prolonged anomaly persists for `rolling_window_size` windows, it may be incorporated into the training set as "normal". The `contamination` parameter partially mitigates this by assuming a fraction of training windows are anomalous.

### 4.4 Feature Space Dynamics

The Drain3 template set grows monotonically as new log patterns are encountered. The feature space therefore expands over time:

| Time | Distinct templates | IF feature space |
|------|--------------------|------------------|
| Bootstrap (t=0) | ~20-40 | 20-40 dims |
| After 1 hour | ~50-100 | 50-100 dims |
| Steady state | ~100-200 | 100-200 dims |

New templates discovered after a model training are excluded from the current model's feature set but recorded in `_all_cluster_ids`. On the next retrain, they are incorporated. By itself the IF would let a burst of entirely new log types — a new deployment, a crash printing unseen stack traces — evade detection until the next retrain cycle. The **novelty signal** (§4.1) closes exactly this gap: it fires on the ratio of logs whose `cluster_id` is not yet in the trained set, so an inrush of unseen templates is flagged immediately, then hands back to the IF/severity signals once the next retrain incorporates them. The **warm-up ramp** prevents the symmetric failure mode — right after a cold (re)start *every* template is unseen, so novelty is damped for the first `novelty_warmup_windows` windows while the baseline matures.

### 4.5 PCA Visualization

After each training, the pipeline fits a 2D PCA projection of the training set for visualization in the web UI:

```python
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(X_train)   # shape (n_windows, 2)
```

Each new scored window is projected into this same 2D space using `pca.transform()`. The scatter plot shows the training distribution and the position of incoming windows, giving operators visual intuition for why a window was flagged as anomalous (outlier in 2D projection).

---

## 5. End-to-End Data Flow

```
K8s API Server
    │
    │  Watch stream (HTTP chunked, real-time push)
    ▼
K8sCollector._event_to_entry()
    │
    │  LogEntry {timestamp, namespace, source, reason, message, raw}
    ▼
LogParser.parse(raw)          ← Drain3 online clustering
    │
    │  ParsedLog {cluster_id, template, raw, namespace, timestamp}
    ▼
WindowBuilder.feed(parsed, timestamp)
    │
    │  WindowData {cluster_counts, raw_logs, namespaces, ...}  [closed every 60s]
    ▼
AnomalyDetector.process(window)
    │
    │  Bootstrap (first 10 windows): accumulate → fit IF → ready
    │  Detection: per-namespace vectorize → max(IF_abs, severity, novelty) → ScoredWindow
    │  Retrain (every 5 windows): fit new IF (per-namespace rows) on last 50 windows
    ▼
ScoredWindow {score ∈ [0,1], is_anomaly, culprit_namespace, pca_x, pca_y}
    │
    ├── score < 0.80 → log to UI, continue
    │
    └── score ≥ 0.80 ─→ AIOPsPipeline._trigger_rca()
                              │
                              ▼
                         OllamaRCA.diagnose(scored_window)
                              │
                              ▼
                         k8s-rca-slm → ROOT CAUSE + KUBECTL
```

---

## 6. Pipeline Configuration

All parameters are centralized in `PipelineConfig` (`config/settings.py`) with environment variable overrides for deployment:

```python
@dataclass
class CollectorConfig:
    namespaces: list[str] | None = None   # None = all namespaces
    window_size_seconds: float = 60.0
    bootstrap_windows: int = 10
    rolling_window_size: int = 50
    retrain_every_n_windows: int = 5

@dataclass
class DetectorConfig:
    anomaly_threshold: float = 0.80
    n_estimators: int = 200
    contamination: float = 0.05
    random_state: int = 42

@dataclass
class DiagnosticsConfig:
    host: str     # OLLAMA_HOST env var, default http://localhost:11434
    model: str    # OLLAMA_MODEL env var, default qwen2.5-coder:1.5b
    enabled: bool = True
    max_logs_in_prompt: int = 40
    timeout_seconds: float = 120.0
```

---

## 7. Design Decisions and Tradeoffs

### 7.1 Fixed Windows vs. Sliding Windows

Fixed non-overlapping windows were chosen over sliding windows for simplicity and to avoid double-counting events. The tradeoff: an anomaly that straddles two window boundaries may appear as two low-score windows rather than one high-score window. The 60-second window size was chosen to be large enough to capture multi-event sequences (CrashLoopBackOff → OOMKilling → BackOff all within seconds) while being small enough to localize the anomaly in time.

### 7.2 L1 Normalization vs. Raw Counts

L1 normalization converts the count vector into a probability distribution over templates. This makes the feature representation volume-invariant: a cluster with 10 events and another with 1000 events in the same proportion of template types produce the same feature vector. The alternative — using raw counts — would cause high-traffic windows to cluster together regardless of their template distribution, reducing detection power.

### 7.3 Drain3 sim_th = 0.4

A lower similarity threshold creates more clusters (higher granularity, more features) but also more noise from spurious splits. A higher threshold merges more templates together, losing discriminative power. `0.4` was selected empirically as a value that produces ~50-150 stable clusters in a mid-sized Kubernetes cluster while keeping semantically distinct patterns separate (e.g., `OOMKilling` and `BackOff` remain in different clusters).

### 7.4 Isolation Forest vs. Alternatives

| Method | Advantage | Disadvantage |
|--------|-----------|--------------|
| Isolation Forest | No labels needed, fast, handles sparse features | Contamination assumption, drift sensitivity |
| OCSVM | Good on low-dimensional data | Slow on high-dimensional sparse vectors |
| Autoencoder | Captures complex patterns | Requires more data, GPU preferred |
| Static rules | Interpretable, zero false negatives for known patterns | No coverage of unknown patterns, requires expert maintenance |

Isolation Forest was chosen for its balance of simplicity, speed, and effectiveness on high-dimensional sparse feature vectors — which is exactly the representation produced by Drain3 template frequency counts. The two classic IF weaknesses are addressed directly: **drift sensitivity** by sliding-window retraining (§4.3) plus the novelty signal and warm-up (§4.4); and the **relative-scoring artifact** (the main cause of false positives) by switching from min-max-within-batch to the **absolute `decision_function`** score (§4.1). Streaming/online IF variants (iForestASD, Robust Random Cut Forest, Half-Space Trees) and Extended Isolation Forest were considered: they improve score *quality* or *online adaptation* but do not address the cold-start / per-namespace-attribution problems that dominated in practice, and add cost on a CPU-only deployment. They remain candidate future work for the IF sub-signal.

### 7.5 Rolling Retraining Strategy

Retraining every 5 windows on the last 50 windows provides adaptation to cluster drift at the cost of computation. The 5-window retrain interval means the model refreshes roughly every 5 minutes (at 60s/window), which is frequent enough to track gradual changes (rolling deployments, traffic growth) without being computationally expensive. scikit-learn's IsolationForest fits in under 100ms on a feature matrix of 50 × 200.

---

## 8. Observed Behavior on Live Cluster

During validation against a real Kubernetes cluster (the same cluster used for dataset generation), the following was observed:

- **Bootstrap time:** ~10 minutes wall-clock (10 × 60s windows)
- **Template saturation:** ~80 distinct Drain3 templates after 30 minutes, growing slowly thereafter
- **Detection latency:** < 60s (one window period) from incident onset to alert
- **Retraining cost:** < 50ms per cycle (50 windows × ~150 features)

**Before vs. after the per-namespace / absolute-scoring hardening (§4):** the early relative-normalization design flagged almost every busy window (~15 namespaces, 200–600 events) at score 1.000 — an unusable false-positive flood. After the change, normal windows score low and *varied* (observed 0.0–0.5) and the only sustained alerts correspond to the cluster's genuine persistent issue (PostgreSQL `role "$(POSTGRES_USER)" does not exist`, severity = 1.00), correctly attributed to the `postgresql` namespace; benign high-volume namespaces (`argocd`, `aeat-retenciones`) no longer fire.

---

## 9. Pending / Future Work

- [ ] Formal precision/recall evaluation with labeled incident injection (chaos engineering)
- [ ] Comparison of Drain3 sim_th values: 0.3, 0.4, 0.5, 0.6 — effect on template count and detection rate
- [ ] Overlapping windows (stride < window_size) to reduce boundary split artifacts
- [ ] Weighted feature vectors: giving higher weight to Warning events vs. Normal events
- [x] Per-namespace anomaly scoring with single-culprit attribution (§4.1, §4.3)
- [x] Absolute IF scoring via `decision_function` — eliminates the relative-normalization false-positive flood (§4.1)
- [x] Novelty signal + warm-up — catches unseen-template bursts and damps the cold-start flood (§4.4)
- [x] Alert deduplication: recurring same-namespace incidents increment a counter instead of new alerts
- [ ] Persistence of trained IF model + Drain3 vocabulary across restarts (currently re-bootstraps on every start)
- [ ] Streaming IF variant (RRCF / Half-Space Trees) or Extended IF for the distribution sub-signal
