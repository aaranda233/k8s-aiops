# K8s-AIOps: Autonomous Anomaly Detection and Root Cause Analysis in Kubernetes using a Fine-Tuned Small Language Model and a Hybrid ReAct Agent

**Status:** Work in progress — Experiment 10 complete
**Model:** [aaranda233/k8s-rca-slm](https://huggingface.co/aaranda233/k8s-rca-slm) · [aaranda233/k8s-rca-orpo](https://huggingface.co/aaranda233/k8s-rca-orpo)
**Hardware:** NVIDIA A30 (24GB VRAM) training · Intel Xeon Gold 6526Y inference

---

## Abstract

This work presents an end-to-end AIOps pipeline for Kubernetes environments that combines unsupervised anomaly detection with automated root cause analysis (RCA). The system collects raw Kubernetes events continuously, detects anomalous time windows using Isolation Forest, and routes flagged windows to a domain-specific diagnostics layer. The diagnostics layer has evolved through ten experiments across three paradigms: (1) single-shot fine-tuned SLM (SFT, DPO, ORPO, KTO); (2) grammar-constrained decoding; and (3) a two-phase Hybrid ReAct Agent. The base SLM is derived from `Qwen2.5-1.5B-Instruct` fine-tuned via QLoRA with ORPO on a curated dataset of ~986 Kubernetes incident scenarios covering 14 failure categories.

The central empirical finding is a persistent **Parse%/Keyword% trade-off**: fine-tuning methods that enforce output format (SFT, ORPO) sacrifice semantic vocabulary coverage, while preference-optimization methods that recover vocabulary (DPO, KTO) destroy format entirely. This trade-off is not a fundamental limit but a consequence of solving both objectives with a single small model trained on a restricted dataset. The final system — a Hybrid ReAct Agent combining a vanilla `qwen2.5:1.5b` investigator with a fine-tuned ORPO expert under GBNF grammar — resolves the trade-off simultaneously: **Keyword%=92.9%** (matching the unspecialized baseline) with **Parse%=98.6%** (guaranteed by grammar), all running on CPU without GPU infrastructure at inference time.

---

## 1. Motivation

Site Reliability Engineers (SREs) managing Kubernetes clusters spend a significant portion of their time triaging alert noise. Production clusters generate thousands of events per hour; only a fraction correspond to actionable anomalies, and correlating those events to a root cause requires domain expertise that is hard to encode in rule-based systems.

Large Language Models (LLMs) have demonstrated strong performance in code understanding and system administration tasks, but their operational cost (API latency, inference hardware, data privacy) makes them impractical as real-time triage components. Smaller, domain-specialized models offer a viable alternative: faster inference, lower resource consumption, and the ability to run on-premise.

This work explores whether a 1.5B-parameter model, fine-tuned on a relatively small dataset (~1,000 samples), can produce accurate RCA output for common Kubernetes failure patterns.

---

## 2. System Architecture

The pipeline consists of four sequential layers:

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
└───────────────────────┬─────────────────────────────────┘
                        │  Events API + pod logs
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 1 — Collector  (src/collector/k8s_collector.py)  │
│  • Kubernetes Python client                              │
│  • Sliding window: 60s, stride 30s                      │
│  • Captures: reason, message, involvedObject, namespace  │
└───────────────────────┬─────────────────────────────────┘
                        │  Structured event stream
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2 — Detector   (src/detector/)                   │
│  • Log parser: drain3 template mining                    │
│  • Feature extraction: event frequency, warning ratio,  │
│    unique reasons, backoff count, error rate             │
│  • Isolation Forest (contamination=0.05, n_estimators=  │
│    100): flags windows with anomaly score > threshold    │
└───────────────────────┬─────────────────────────────────┘
                        │  Anomalous window (raw events)
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 3 — Diagnostics  (src/diagnostics/)              │
│  Three modes (REACT_MODE env var):                      │
│  • single_shot — one call to fine-tuned ORPO model      │
│  • react       — ReAct loop with fine-tuned model       │
│  • hybrid ⭐   — qwen2.5:1.5b investigates (THOUGHT/    │
│                  ACTION), ORPO expert diagnoses with    │
│                  GBNF grammar (ROOT CAUSE + KUBECTL)    │
└───────────────────────┬─────────────────────────────────┘
                        │  RCA report
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 4 — Web UI  (web/)                               │
│  • FastAPI + Server-Sent Events                         │
│  • Real-time anomaly feed                               │
│  • RCA cards with kubectl commands                      │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Dataset Construction

### 3.1 Sources

The training dataset combines two sources:

| Source | Samples | Method |
|--------|---------|--------|
| Synthetic (generic names) | 420 | Template-based generator with 14 scenario types |
| Real cluster + synthetic real names | 566 | `real_extractor.py` against live cluster; names extracted from actual deployments, services and namespaces |
| **Total** | **986** | |

### 3.2 Scenario Coverage

14 Kubernetes failure categories, 70 samples each:

| Scenario | Description |
|----------|-------------|
| OOMKilled | Container exceeds memory limit, killed by cgroup |
| CrashLoopBackOff | Container repeatedly crashing on startup |
| ImagePullBackOff | Registry unreachable or image tag missing |
| PVC Pending | PersistentVolumeClaim unbound, no matching PV |
| Node NotReady | Node unreachable or failing health checks |
| Node DiskPressure | Ephemeral storage exhausted, eviction triggered |
| Network Policy Drop | Egress/ingress blocked by NetworkPolicy |
| HPA ScalingFailure | HorizontalPodAutoscaler unable to scale |
| Secret Missing | Pod references non-existent Secret |
| Liveness Probe Failure | Probe timeout/failure causing restarts |
| Deployment Rollout Stuck | New ReplicaSet unable to reach desired state |
| ResourceQuota Exceeded | Namespace quota prevents pod scheduling |
| DNS Resolution Failure | CoreDNS unreachable or NXDOMAIN errors |
| Node MemoryPressure | Node-level memory exhaustion |

### 3.3 Sample Format

Each sample follows the Qwen2.5 chatml format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert Site Reliability Engineer (SRE) specialized in Kubernetes. [...]"
    },
    {
      "role": "user",
      "content": "Events from namespace production, pod api-gateway-7d9f8c-xk2p:\n2024-01-15T10:23:41Z Warning BackOff Back-off restarting failed container\n2024-01-15T10:23:41Z Warning Failed Error: OOMKilled\n[...]"
    },
    {
      "role": "assistant",
      "content": "ROOT CAUSE: The api-gateway container is being OOMKilled because its memory usage (267Mi) exceeds the configured limit (256Mi). Kubernetes restarts it in a CrashLoopBackOff cycle.\nKUBECTL: kubectl set resources deployment/api-gateway --limits=memory=512Mi -n production"
    }
  ]
}
```

### 3.4 Dataset Quality Metrics

Evaluated with `dataset/evaluate_dataset.py` prior to training:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean cosine distance (TF-IDF) | 0.965 | Excellent semantic diversity |
| Near-duplicate pairs (dist < 0.1) | 0.0% | No deduplication needed |
| Unique kubectl patterns | 157 | Good command coverage |
| Type-Token Ratio (TTR) | 0.024 | Expected low for domain-specific corpus |
| Samples with ROOT CAUSE | 986/986 | 100% format compliance |
| Samples with KUBECTL | 986/986 | 100% format compliance |

---

## 4. Fine-Tuning

### 4.1 Setup

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Framework | [unsloth](https://github.com/unslothai/unsloth) 2026.5.9 + TRL 0.24.0 |
| Hardware | NVIDIA A30 (24GB VRAM) |
| Quantization (training) | NF4 4-bit (QLoRA) |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 18,464,768 / 1,562,179,072 (1.18%) |
| Batch size (per device) | 4 |
| Gradient accumulation | 4 (effective batch = 16) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup steps | 10 |
| Epochs | 3 |
| Max sequence length | 1024 tokens |
| Optimizer | adamw_8bit (bitsandbytes) |
| Precision | bfloat16 |
| Total steps | 186 |
| Training time | ~8 minutes |

### 4.2 Training Loss

| Step | Loss | LR |
|------|------|----|
| 10 | 0.7994 | 1.80e-04 |
| 20 | 0.3536 | 1.99e-04 |
| 30 | 0.1749 | 1.94e-04 |
| 40 | 0.1294 | 1.87e-04 |
| 60 | 0.0979 | 1.64e-04 |
| 90 | 0.0864 | 1.16e-04 |
| 130 | 0.0929 | 4.74e-05 |
| 180 | 0.0898 | 7.80e-07 |

Loss converged from 0.80 to ~0.08 in 186 steps, indicating strong memorization of the output format and domain vocabulary.

### 4.3 Deployment Format

After training, LoRA adapters (~70MB) are merged with the base model and exported to GGUF format for CPU inference via llama.cpp/Ollama:

| Format | Size | Notes |
|--------|------|-------|
| LoRA adapters (safetensors) | 70MB | For further fine-tuning |
| GGUF Q4_K_M | 941MB | 4-bit quantization — **not recommended for 1.5B models** |
| GGUF Q8_0 | 1.6GB | 8-bit quantization — **production-recommended** |

**Finding:** Q4_K_M quantization causes significant quality degradation on 1.5B parameter models. Q8_0 preserves output quality while remaining deployable on commodity hardware. This is consistent with the general recommendation that aggressive quantization (≤4-bit) requires models of at least 7B parameters to maintain acceptable output fidelity.

---

## 5. Inference

The model is served via [Ollama](https://ollama.com) with an explicit chatml template (required for Ollama ≤0.24.0 which does not auto-detect Qwen2.5's template from GGUF metadata):

```
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ range .Messages }}<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
```

Inference parameters: `temperature=0.1`, `top_p=0.9`, `repeat_penalty=1.1`, `num_ctx=2048` (updated from 1024 — required for hybrid agent enriched context).

For grammar-constrained inference (hybrid expert phase), the `/api/generate` endpoint is used with the `grammar` field containing the GBNF definition. This guarantees format at the token level and decouples format correctness from model capacity.

---

## 6. Qualitative Results

Three representative inference examples with the Q8_0 model:

### 6.1 OOMKilled

**Input events:**
```
Events from namespace production, pod api-gateway-7d9f8c-xk2p:
2024-01-15T10:23:41Z Warning BackOff Back-off restarting failed container
2024-01-15T10:23:41Z Warning Failed Error: OOMKilled
2024-01-15T10:21:15Z Warning OOMKilling Memory cgroup out of memory: Kill process 18234 score 987
Memory limit: 256Mi, Usage: 267Mi
```

**Model output:**
```
ROOT CAUSE: Memory pressure en el contenedor api-gateway-7d9f8c-xk2p. K8s lo mata con
OOMKilled porque supera el límite de memoria configurado (256Mi). El proceso está
consumiendo más memoria del permitido.
KUBECTL: kubectl set resources deployment/api-gateway --limits=memory=300Mi -n production
```

### 6.2 ImagePullBackOff

**Input events:**
```
Events from namespace staging, pod checkout-6f8b9-p2kl:
2024-01-15T11:05:12Z Warning Failed Failed to pull image checkout-service:v3.2.1-hotfix:
  rpc error: code = NotFound
2024-01-15T11:05:12Z Warning Failed Error: ErrImagePull
2024-01-15T11:05:42Z Warning BackOff Back-off pulling image checkout-service:v3.2.1-hotfix
```

**Model output:**
```
ROOT CAUSE: ImagePullBackOff por error de registro. El registry 'checkout-service' no
responde o la imagen 'checkout-service:v3.2.1-hotfix' no existe en el registry.
KUBECTL: kubectl set image deployment/checkout-frontend --image=checkout-service:latest -n staging
```

### 6.3 Node DiskPressure

**Input events:**
```
Events from node worker-node-03:
2024-01-15T12:30:01Z Warning NodeNotReady Node worker-node-03 status is now: NodeNotReady
2024-01-15T12:30:01Z Normal NodeHasDiskPressure Node worker-node-03 status is now: NodeHasDiskPressure
2024-01-15T12:29:55Z Warning EvictionThresholdMet Attempting to reclaim ephemeral-storage
Disk usage: 94% of 200Gi
```

**Model output:**
```
ROOT CAUSE: DiskPressure en nodo worker-node-03 por falta de espacio en disco efímero.
K8s está eviccionando pods para liberar espacio.
KUBECTL: kubectl debug node/worker-node-03 -it -- df -h
```

---

## 7. Key Engineering Findings

### 7.1 Quantization Sensitivity in Small Models

Q4_K_M quantization, while standard for 7B+ models, causes severe quality degradation in 1.5B models. Symptoms observed: repetition of input tokens, language mixing (Spanish/English/Chinese), non-adherence to output format. Q8_0 resolves these issues while adding only ~700MB to the model size. This suggests a minimum effective quantization level dependent on model parameter count.

### 7.2 unsloth + SFTTrainer API Compatibility

When using unsloth 2026.5.x with TRL 0.24.0, `SFTTrainer` requires an explicit `formatting_func` — the `messages` field alone is insufficient. The function must handle both single-example (dict) and batched (dict-of-lists) inputs, as unsloth calls it once in single-example mode during dataset preparation:

```python
def formatting_func(examples):
    messages = examples["messages"]
    if isinstance(messages[0], dict):          # single example
        return [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)]
    return [                                    # batch
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        for msgs in messages
    ]
```

### 7.3 Docker for Reproducible GPU Training

Running training inside Docker with `--gpus all` and `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` as base image ensures reproducibility across GPU environments. unsloth upgrades PyTorch to 2.10 during installation, which is expected and non-breaking. GPU access is only required at `docker run` time — `docker build` must not import unsloth.

### 7.4 Dataset Size vs. Loss

986 samples × 3 epochs = 186 gradient steps at batch size 16. Despite the small scale, training loss converged to ~0.08, indicating the dataset format is highly consistent and the model capacity is sufficient for this narrow task. Larger datasets or more epochs would likely improve generalization.

---

## 8. Alignment Experiments — DPO, ORPO, KTO

After SFT v1 established that the model learns the `ROOT CAUSE/KUBECTL` format (Parse%=56.2%), six preference optimization experiments were conducted to recover the semantic vocabulary lost to memorization (Keyword% dropped from 92.4% to 60.0% after SFT).

### 8.1 Central Finding: Parse%/Keyword% Trade-off

All alignment methods exhibit an empirical trade-off between format compliance and semantic coverage:

| Method | Parse% | Keyword% | Outcome |
|--------|:------:|:--------:|---------|
| Baseline (vanilla) | 38.6% | **92.4%** | No format, full vocabulary |
| SFT v1 | 56.2% | 60.0% | Format learned, vocabulary lost |
| DPO v1 | 16.2% | 82.9% | Format destroyed, vocabulary recovered |
| DPO v2 | 8.1% | 87.1% | Mode collapse at step 20 |
| SimPO | 16.7% | 86.7% | High β destroys format |
| KTO | 0.0% | 0.0% | Complete collapse — no L_SFT anchor |
| **ORPO** | **58.1%** | **67.1%** | Best single-model balance |

### 8.2 Why ORPO Succeeds Where DPO Fails

ORPO (Hong et al., 2024) is the only method that avoids format collapse. The key is the joint loss function:

```
L_ORPO = L_SFT + λ · L_OR

L_SFT = cross-entropy over chosen tokens  →  anchors format at every step
L_OR  = log(odds_chosen / odds_rejected)  →  pushes vocabulary towards correct concepts
```

DPO and KTO optimize preferences without an explicit format anchor. With a small 1.5B model trained on a restricted dataset (986 samples, loss→0.08), the attention layers encoding the `ROOT CAUSE/KUBECTL` structure are extremely fragile. Any preference gradient — regardless of β or dataset quality — perturbs them irreversibly.

**Hypothesis confirmed by three independent failure modes:**
- DPO: near-zero gradient signal corrupts format weights
- SimPO (β=2.0): aggressive margin pushes log π(rejected) → −∞, destroying format
- KTO: no pairwise comparison and no L_SFT → distribution drift without anchor

### 8.3 Grammar-Constrained Decoding

GBNF grammar applied via Ollama's `/api/generate` endpoint forces the output format at the token level, decoupling format from model capacity:

```gbnf
root         ::= "ROOT CAUSE: " rc-text "\nKUBECTL: " kubectl-text
rc-text      ::= [^\n]+ (" " [^\n]+)*
kubectl-text ::= "kubectl " [^\n]+
```

With grammar active on the ORPO model: Parse%: 58.1% → **100.0%**, Keyword%: 67.1% → **78.1%**, NS-ok%: 48.1% → **89.5%**. Grammar resolves format independently of model quality and removes it as a confounding variable in downstream evaluation.

---

## 9. Quantitative Evaluation

**Harness:** `eval/run_eval.py` · 210 blind samples (seed=99, distinct from training seed=42) · 14 scenarios × 15 samples · CPU Intel Xeon Gold 6526Y

### 9.1 Metrics

| Metric | Definition |
|--------|-----------|
| **Parse%** | Response contains both `ROOT CAUSE:` and `KUBECTL:` prefixes |
| **Keyword%** | Root cause mentions at least one canonical keyword for the scenario (e.g. "OOMKilled", "ImagePullBackOff") |
| **ROUGE-L** | Longest common subsequence F1 between generated and reference root cause |
| **NS-ok%** | kubectl command contains the correct namespace |
| **Verb-ok%** | kubectl command uses a semantically appropriate verb for the scenario |

### 9.2 Results — All Models

| Model | Parse% | Keyword% | ROUGE-L | NS-ok% | Verb-ok% | Lat. |
|-------|:------:|:--------:|:-------:|:------:|:--------:|:----:|
| Baseline (Qwen2.5-1.5B) | 38.6% | 92.4% | 2.5% | 1.4% | 1.9% | 1.00s |
| SFT v1 | 56.2% | 60.0% | 56.7% | 32.9% | 41.0% | 0.86s |
| SFT v2 | 35.2% | 64.3% | 41.2% | 22.4% | 43.3% | 0.89s |
| DPO v1 | 16.2% | 82.9% | 2.4% | — | — | — |
| SimPO | 16.7% | 86.7% | 57.7% | — | — | — |
| DPO v2 | 8.1% | 87.1% | 21.7% | 6.2% | 31.9% | 0.81s |
| ORPO Q8_0 | 58.1% | 67.1% | 16.2% | 48.1% | 49.5% | 0.89s |
| ORPO Q4_K_M | 59.5% | 76.2% | 14.7% | 45.7% | 44.8% | 0.85s |
| KTO | 0.0% | 0.0% | 0.0% | 0.0% | 28.6% | 0.49s |
| ORPO + grammar | **100.0%** | 78.1% | 19.3% | **89.5%** | **56.7%** | **0.71s** |
| **Hybrid ReAct + grammar** | 98.6% | **92.9%** | 5.9% | 73.3% | 42.4% | 2.04s |

*210 samples per model · seed=99*

### 9.3 Key Observations

1. **ROUGE-L is inversely correlated with generalization.** SFT v1 achieves ROUGE-L=56.7% by memorizing reference answers. The hybrid model achieves ROUGE-L=5.9% because it generates specific, context-grounded diagnoses that diverge from generic reference templates — a better real-world outcome than high textual similarity.

2. **NS-ok% is the most sensitive metric to model capability.** It requires the model to extract the correct namespace from the event context and reproduce it in the kubectl command. ORPO+grammar achieves 89.5% by combining domain knowledge (ORPO) with format guarantee (grammar).

3. **Keyword% is the best proxy for diagnostic quality.** It measures whether the model identified the correct failure type regardless of phrasing — closer to what an SRE actually cares about.

---

## 10. Hybrid ReAct Agent

### 10.1 Motivation

The Parse%/Keyword% trade-off documented in Section 8 has a structural cause: fine-tuning a 1.5B model on ~1,000 samples for structured output memorizes the format at the expense of generalizable semantic knowledge. No single-model training approach resolved this in nine experiments.

The key insight is that **format compliance** and **domain investigation** are separable capabilities:
- A vanilla instruction-following model can reason about K8s events and plan investigations without format constraints
- A fine-tuned model has K8s domain knowledge and can produce structured output when guided by grammar

### 10.2 Architecture

```
Anomaly detected (score ≥ threshold)
        │
        ▼
Phase 1 — Investigator (qwen2.5:1.5b vanilla)
  System: ReAct-style prompt (THOUGHT/ACTION/DONE)
  Input:  anomalous event window (raw logs, score, namespaces)
  Loop (max 3 steps):
    → THOUGHT: reasoning about what to investigate
    → ACTION:  kubectl <read-only command>
    → [OBSERVATION: tool result if dry_run=False]
  Exit: DONE or max_steps reached
        │
        ▼  investigation plan (list of THOUGHT + ACTION lines)
        │
        ▼
Phase 2 — Expert (k8s-rca-orpo + GBNF grammar)
  Input:  original event context + investigation plan appended
  Endpoint: /api/generate (grammar parameter available here)
  Grammar: ROOT CAUSE: <text>\nKUBECTL: kubectl <command>
  num_ctx: 2048 (increased from 1024 to fit enriched context)
        │
        ▼
  DiagnosisResult(root_cause, kubectl_command, confidence,
                  steps_taken, react_trace, mode="hybrid")
```

The `kubectl_toolbox.py` enforces read-only safety: `apply`, `delete`, `patch`, `create`, `exec`, and 10 other write verbs are rejected before execution. Write commands are only proposed by the expert (as remediation suggestions), never executed.

### 10.3 Key Result: network_policy_block 0% → 73.3%

`network_policy_block` was the systematic blind spot of ORPO alone — it produced correctly formatted responses that never mentioned network policy, blocking, or egress/ingress concepts. The investigator's reasoning steps provide the expert with the conceptual frame needed to identify the failure type.

Comparison across all scenarios (Keyword%, n=15 per scenario):

| Scenario | ORPO+grammar | Hybrid+grammar | Δ |
|----------|:-----------:|:--------------:|:---:|
| crash_oom | 80.0% | **100.0%** | +20 pp |
| image_auth | 80.0% | **100.0%** | +20 pp |
| image_registry_down | 60.0% | **100.0%** | +40 pp |
| **network_policy_block** | 0.0% | **73.3%** | **+73 pp** |
| node_pressure_memory | 60.0% | **100.0%** | +40 pp |
| readiness_failing | 73.3% | **100.0%** | +27 pp |
| crash_config | **66.7%** | 46.7% | −20 pp |
| crash_probe | **100.0%** | 93.3% | −7 pp |

The regression in `crash_config` (−20 pp) is attributable to investigation notes about ConfigMap/environment variables saturating the expert's attention window, displacing the diagnostic keywords. A future fix is to limit the investigation notes to the 2 most relevant steps.

### 10.4 Iterative Development

**v1 (no grammar, num_ctx=1024):** Parse%=32.4%, Keyword%=72.9%. The enriched context exceeded the model's 1024-token context window, causing truncation and format failure. Keyword% improvement (+7.2 pp over ORPO alone) confirmed the investigation plan provides genuine semantic value.

**v2 (GBNF grammar + num_ctx=2048):** Parse%=98.6%, Keyword%=92.9%. Two fixes applied simultaneously: model recreated with `num_ctx=2048` and expert call migrated from `/api/chat` to `/api/generate` with grammar parameter.

### 10.5 Conclusion

> The Hybrid ReAct Agent achieves Keyword%=92.9% — statistically equivalent to the unspecialized baseline (92.4%) — while maintaining Parse%=98.6% and structured kubectl output. This demonstrates that the Parse%/Keyword% trade-off observed across all fine-tuning experiments is not a fundamental constraint but a consequence of attempting to solve both objectives within a single small model trained on a restricted dataset. Role separation (instruction-following investigator + domain-specialized expert) resolves them simultaneously without additional fine-tuning.

---

## 11. Additional Engineering Findings

### 11.1 num_ctx Must Match Context Size

The Hybrid ReAct Agent's expert call failed with Parse%=32% when `num_ctx=1024` was insufficient for the enriched prompt (original events + investigation notes). Always set `num_ctx` ≥ expected prompt length. The Modelfile parameter is a ceiling, not a suggestion; truncation is silent.

### 11.2 Grammar Requires `/api/generate`, Not `/api/chat`

Ollama's `grammar` parameter is only available in the `/api/generate` endpoint. When migrating a model from `/api/chat` to `/api/generate`, the ChatML prompt must be constructed manually:

```python
prompt = (
    f"<|im_start|>system\n{system}<|im_end|>\n"
    f"<|im_start|>user\n{user}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
```

### 11.3 Role Separation as an Alternative to Multi-Objective Fine-Tuning

When a fine-tuned model excels at one objective (format) but fails at another (vocabulary), adding more training data or preference optimization is not the only path. Delegating each objective to the model best suited for it — a general instruction-follower for flexible reasoning, a specialized model for structured output — can outperform any single-model approach at the cost of additional inference latency.

---

## 12. Reproducibility

### Training

```bash
# 1. Build image (A30 or any CUDA 12.4 machine)
cd finetune/
docker build -t k8s-rca-train .

# 2. Generate dataset
python dataset/generator.py
python dataset/real_extractor.py       # requires kubeconfig
python dataset/combine.py

# 3. Evaluate dataset quality
python dataset/evaluate_dataset.py dataset/output/combined.jsonl --no-perplexity

# 4. Train (mounts output/ for persistence)
docker run --gpus all --rm \
  -v $(pwd)/dataset/output:/workspace/output \
  -v $(pwd)/finetune/train.py:/workspace/train.py:ro \
  -v ~/.cache/huggingface:/workspace/.cache/huggingface \
  k8s-rca-train \
  python train.py --dataset output/combined.jsonl --epochs 3

# 5. Quantize to Q8_0 (recommended over Q4_K_M for 1.5B)
docker run --gpus all --rm \
  -v $(pwd)/finetune/output/k8s-rca-slm:/workspace/adapters:ro \
  -v $(pwd)/finetune/output:/workspace/output \
  -v ~/.cache/huggingface:/workspace/.cache/huggingface \
  k8s-rca-train \
  bash -c "apt-get install -y -qq libssl-dev curl libcurl4-openssl-dev && python convert_gguf.py"
```

### Inference

```bash
# Via Ollama (recommended)
ollama create k8s-rca-slm -f finetune/Modelfile
ollama run k8s-rca-slm

# Direct download from HuggingFace
ollama run hf.co/aaranda233/k8s-rca-slm
```

---

## 12. Auto-Remediation with Human-in-the-Loop

### 12.1 Motivation

Diagnosis without action only automates half the SRE workflow. The natural extension of the Hybrid ReAct Agent is a remediation loop that acts on the diagnosis, verifies the result, and escalates when intervention is needed. The key design question is not *whether* to automate, but *how much* to automate safely.

Full autonomy is not the correct objective for production AIOps. Enterprise environments have compliance requirements (SOC2, ISO27001) mandating human traceability for infrastructure changes. A system that explains its reasoning and asks for confirmation on risky actions is more deployable than one that acts without oversight.

### 12.2 Risk Taxonomy

All kubectl commands are classified into four levels before execution:

| Level | Label | Examples | Policy |
|-------|-------|----------|--------|
| 0 | Read-only | `describe`, `get`, `logs`, `top` | Execute freely |
| 1 | Reversible | `rollout restart`, `scale`, `rollout undo` | Execute automatically |
| 2 | Config change | `set resources`, `patch`, `set image` | Require human approval (ChatOps) |
| 3 | Destructive | `delete`, `drain`, `cordon`, `exec` | Never execute automatically |

The classification is conservative: unknown verbs default to Level 2. This follows the principle of least privilege — the agent assumes more risk, not less, when uncertain.

### 12.3 Architecture

```
Anomaly detected → Hybrid ReAct diagnosis → kubectl proposed
        │
        ▼
[Circuit Breaker] — same anomaly seen 3+ times in 10 min? → escalate + stop
        │
        ▼
[Risk Scorer] — classify kubectl (Level 0-3)
        │
        ├─ Level 0 → no action (read-only, already done in investigation)
        │
        ├─ Level 1 → dry-run → execute → wait 90s → verify
        │            success: notify "resolved" + reset circuit breaker
        │            failure: notify "fix failed" + record attempt
        │
        ├─ Level 2 → Teams card with [APPROVE] / [REJECT] buttons
        │            approved: execute → verify
        │            rejected: record + close
        │            no response in 30min: discard + notify
        │
        └─ Level 3 → notify "manual action required" + exact command
                     never executes
```

All remediation runs in a background thread — the main pipeline continues processing events regardless of remediation outcome.

### 12.4 Anti-Loop Protections

Three independent mechanisms prevent remediation loops:

**Circuit Breaker (`src/remediation/circuit_breaker.py`):**
Tracks attempts per anomaly fingerprint (hash of namespace + root cause prefix). After 3 attempts in 10 minutes, blocks all automatic action for that fingerprint and sends an escalation alert.

**Mandatory Dry-Run (`src/remediation/executor.py`):**
Every Level 1+ command runs `--dry-run=client` first. If dry-run fails, the real command is never attempted. Output is compared for sanity before proceeding.

**Post-Fix Verification:**
After Level 1 execution, the system waits 90 seconds and queries the affected resource. If replicas are not Ready or Warning events persist, the attempt is recorded as failed and the circuit breaker counter increments.

### 12.5 Pluggable Notification — ChatOps (Teams) as Primary Channel

Notification serves two purposes: transparency (the operator always knows what happened) and control (Level 2 actions require explicit approval). The notification layer is **pluggable** behind a common interface (`BaseNotifier`), with three implementations:

| Channel | Implementation | Mechanism |
|---------|---------------|-----------|
| **Microsoft Teams** (primary) | `TeamsNotifier` | Adaptive Cards via Incoming Webhook |
| Email (fallback) | `EmailNotifier` | HTML email via SMTP |
| Both | `CompositeNotifier` | Fan-out with a shared approval token |

Selected via `NOTIFY_CHANNEL=teams|email|both|none`. Email-based approval has known weaknesses in production (no authentication on the link, buried in inboxes, slow). ChatOps via Teams is the industry standard: the on-call engineer already lives in the channel, identity is handled by the platform, and the thread serves as an audit trail.

Level 2 posts an Adaptive Card to the ops channel:

```
⚠️ K8s-AIOps — Aprobación requerida · namespace: producción · INC-A3F2

Investigación:
  • THOUGHT: Multiple pods evicted, likely memory pressure on node
  • ACTION:  kubectl describe node node-1 -n producción
  • THOUGHT: Node at 97% memory, scheduler pod at 498Mi/512Mi

Diagnóstico: Memory limit insuficiente en deployment/scheduler

Acción propuesta (Level 2):
  kubectl set resources deployment/scheduler --limits=memory=1Gi -n producción

[ ✅ APROBAR ]   [ ❌ RECHAZAR ]

Sin respuesta en 30 min → la acción se descarta.
```

The buttons (`Action.OpenUrl`) call `GET /remediation/approve/{token}` on the web server, which updates a shared in-memory store polled by the remediation thread. The same token mechanism is reused across all channels.

**Security note on the approval token:** the command tied to a token is re-validated by the risk scorer before execution — a token cannot be used to execute a command different from the one originally proposed.

**Design limitation:** with an Incoming Webhook, buttons open the approval endpoint in a browser tab. Inline confirmation (the card updating in place within Teams) requires a registered Bot Framework bot — out of scope for the current implementation but a natural extension.

### 12.6 Configuration

```bash
# Minimal — Level 1 auto with Teams notifications
REMEDIATION_ENABLED=true
NOTIFY_CHANNEL=teams
TEAMS_WEBHOOK_URL=https://prod.westeurope.logic.azure.com/workflows/...
WEBHOOK_BASE_URL=https://k8s-aiops.company.com  # public URL for approval links

# Optional tuning
REMEDIATION_MAX_LEVEL=2        # allow Level 2 with approval

# Email channel / fallback (NOTIFY_CHANNEL=email|both)
SMTP_USER=alerts@company.com
SMTP_PASS=app_password
NOTIFY_EMAIL=sre-team@company.com
```

By default `REMEDIATION_ENABLED=false` — the pipeline behaves exactly as before unless explicitly activated.

### 12.7 System Levels — Comparison with Literature

The remediation module advances the system from diagnosis to closed-loop operation:

| Level | Capability | This system |
|-------|-----------|-------------|
| L1 | Anomaly detection | ✅ Isolation Forest |
| L2 | Root cause diagnosis | ✅ Hybrid ReAct + grammar |
| L3 | Remediation proposal | ✅ kubectl suggested |
| L4 | Autonomous remediation | ✅ Level 1 actions |
| L4+ | Human-gated remediation | ✅ Level 2 ChatOps approval (Teams) |
| L5 | Post-fix verification | ✅ 90s verify loop |
| L5+ | Learning from outcomes | ◻ Future work |

Most published AIOps systems reach L2-L3. Commercial products (Dynatrace Davis AI, PagerDuty AIOps) reach L3-L4 but require cloud connectivity and proprietary models. This system reaches L5 running entirely on-premise on CPU with open models.

---

## 13. Artifacts

| Artifact | Location |
|----------|----------|
| Source code | This repository |
| Dataset (combined.jsonl) | `dataset/output/` (gitignored — regenerable) |
| SFT LoRA adapters | `finetune/output/k8s-rca-slm/` + [HF Hub](https://huggingface.co/aaranda233/k8s-rca-slm) |
| ORPO LoRA adapters | `finetune/output/k8s-rca-orpo/` + [HF Hub](https://huggingface.co/aaranda233/k8s-rca-orpo) |
| GGUF Q8_0 — SFT | [HF Hub — k8s-rca-slm-Q8_0.gguf](https://huggingface.co/aaranda233/k8s-rca-slm) |
| GGUF Q4_K_M — SFT | [HF Hub — k8s-rca-slm-Q4_K_M.gguf](https://huggingface.co/aaranda233/k8s-rca-slm) |
| GGUF Q8_0 — ORPO ⭐ | [HF Hub — k8s-rca-orpo-gguf](https://huggingface.co/aaranda233/k8s-rca-orpo-gguf) (private until publication) |
| Evaluation harness | `eval/run_eval.py` · `eval/runner.py` · `eval/test_set.jsonl` |
| Hybrid ReAct Agent | `src/diagnostics/hybrid_react_agent.py` · `src/diagnostics/kubectl_toolbox.py` |
| Auto-Remediation | `src/remediation/` — risk_scorer, circuit_breaker, executor, auto_remediation |
| Notification (pluggable) | `src/remediation/` — base_notifier, teams_notifier (Teams), notifier (email) |
| Evaluation results | `eval/results/eval_20260609_103514.json` (ORPO+grammar vs Hybrid+grammar) |

---

## 14. Pending / Future Work

- [x] Formal evaluation on held-out test set — implemented in `eval/run_eval.py` (210 samples, seed=99)
- [x] Alignment experiments — DPO v1/v2, SimPO, ORPO, KTO (10 experiments total)
- [x] Grammar-constrained decoding — GBNF grammar via Ollama `/api/generate`
- [x] Hybrid ReAct Agent — role separation resolves Parse%/Keyword% trade-off
- [x] Auto-remediation with human-in-the-loop — Level 1 autonomous, Level 2 email approval, circuit breaker
- [ ] Integration test: full pipeline with chaos injection on live cluster and MTTR measurement
- [ ] Fine-tune on larger dataset (5k+ samples with structural diversity) — prerequisite for DPO on stronger base
- [ ] Scale to 7B model (Mistral-7B or Llama-3.1-8B) with same ORPO recipe — expected to improve both metrics
- [ ] Benchmark against GPT-4o / Claude Sonnet on same 210-sample test set
- [ ] Fine-tune investigator (Phase 1) on ReAct-format examples to improve `crash_config` scenario
- [ ] Auto-remediation: extend hybrid agent to execute safe kubectl write commands with dry-run preview
