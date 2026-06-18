# K8s-AIOps: Autonomous Anomaly Detection and Root Cause Analysis in Kubernetes using a Fine-Tuned Small Language Model and a Hybrid ReAct Agent

**Status:** Work in progress — Experiment 10 complete
**Model:** [aaranda233/k8s-rca-slm](https://huggingface.co/aaranda233/k8s-rca-slm) · [aaranda233/k8s-rca-orpo](https://huggingface.co/aaranda233/k8s-rca-orpo)
**Hardware:** NVIDIA A30 (24GB VRAM) training · Intel Xeon Gold 6526Y inference

---

## Abstract

This work presents an end-to-end AIOps pipeline for Kubernetes environments that combines unsupervised anomaly detection with automated root cause analysis (RCA). The system collects raw Kubernetes events continuously, detects anomalous time windows using Isolation Forest, and routes flagged windows to a domain-specific diagnostics layer. The diagnostics layer has evolved through ten experiments across three paradigms: (1) single-shot fine-tuned SLM (SFT, DPO, ORPO, KTO); (2) grammar-constrained decoding; and (3) a two-phase Hybrid ReAct Agent. The base SLM is derived from `Qwen2.5-1.5B-Instruct` fine-tuned via QLoRA with ORPO on a curated dataset of ~986 Kubernetes incident scenarios covering 14 failure categories.

The central empirical finding is a persistent **Parse%/Keyword% trade-off**: fine-tuning methods that enforce output format (SFT, ORPO) sacrifice semantic vocabulary coverage, while preference-optimization methods that recover vocabulary (DPO, KTO) destroy format entirely. This trade-off is not a fundamental limit but a consequence of solving both objectives with a single small model trained on a restricted dataset. The final system — a Hybrid ReAct Agent combining a vanilla `qwen2.5:1.5b` investigator with a fine-tuned ORPO expert under GBNF grammar — resolves the trade-off simultaneously: **Keyword%=92.9%** (matching the unspecialized baseline) with **Parse%=98.6%** (guaranteed by grammar), all running on CPU without GPU infrastructure at inference time.

Beyond detection and diagnosis, the system extends to a full **operational console** built entirely on read-only access to the Kubernetes API (no observability infrastructure required): dual detection sources (control-plane events + application logs), auto-remediation with human-in-the-loop approval via ChatOps (Microsoft Teams), a conversational read-only investigation agent ("chat with the cluster"), a live topology view, and a rule-based security posture scanner. The same investigation pipeline that diagnoses operational anomalies thus also audits security posture — all without GPU at inference time and without deploying agents into the cluster.

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

## 13. Auto-Remediation with Human-in-the-Loop

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

### 12.8 Reproducible Demonstration of Both Remediation Modes

To demonstrate human-approval and automatic remediation deterministically — without waiting for a natural anomaly whose proposed command happens to be Level 1 — a demo trigger is provided, gated behind `AIOPS_DEMO=true`:

```
POST /api/demo/incident?mode=auto    # shadow OFF: executes + verifies, no human
POST /api/demo/incident?mode=human   # shadow ON: stays pending_approval in /incidents
```

It injects a synthetic Level 1 incident (`rollout restart`) through the **real** remediation path, sharing the console's incident store, targeting a disposable `nginx-demo` deployment in an `aiops-demo` namespace. The operator then approves/rejects in the web console (or via `POST /api/incidents/{id}/{approve,reject}`):

- **Automatic** — the rollout restart executes immediately; verification confirms the new pod is ready → `resolved`.
- **Approve** — the incident waits in `pending_approval`; on approval the restart executes and verifies → `resolved`.
- **Reject** — the incident is marked `rejected`; nothing is executed (the pod is untouched).

This validation surfaced a real executor bug: `kubectl rollout restart` does not accept `--dry-run`, so the blanket dry-run gate silently blocked the most common Level 1 action. The executor now skips the dry-run gate only for commands that cannot support it (`rollout restart`/`undo`), which are already reversible and risk-classified.

---

## 14. Operational Console and Extended Capabilities

Beyond the detection→diagnosis→remediation core, the system grew into a full operational console. Every capability below uses **only read-only Kubernetes API access** (native Python client for data gathering; the `kubectl` CLI, whitelisted to read-only verbs, for the chat and remediation layers). No Loki, Prometheus, or agents are deployed — the system stays portable and infrastructure-free.

### 14.1 Dual Detection Source — Events + Application Logs

Detection originally consumed only Kubernetes **Events** (control-plane signals: `Pulled`, `BackOff`, `OOMKilling`). Events are sparse — a healthy cluster emits very few. A second source was added: **application logs** (`LogCollector`, `read_namespaced_pod_log`), feeding the same Drain3 + Isolation Forest pipeline. This captures app-level signal (errors, stack traces, failed init) that never surfaces as a K8s event.

Design constraints (safety-first): read-only, bounded by `since_seconds`/`tail_lines`/`max_pods`, polling (not N persistent streams), opt-in, cluster-wide or scoped to namespaces. On the production cluster, enabling logs raised per-window signal from 1-2 events to 600+ lines across 34 namespaces. Known limitation: `tail_lines` under-samples very high-volume pods — mitigated by tuning, or by swapping the source for a Loki-backed collector (the layer-1 source is pluggable; deliberately not adopted, as it adds infrastructure for capabilities the real-time use case does not require).

### 14.2 Web Console — Five Views

A FastAPI + WebSocket console exposes the system through five navigable views:

| View | Purpose |
|------|---------|
| **Dashboard** | Watch the algorithm live: Drain3 templates, Isolation Forest PCA, scored windows |
| **Incidencias** | Operations inbox: incidents with diagnosis + proposed kubectl, approve/reject |
| **Chat** | Conversational read-only investigation of the cluster |
| **Topología** | Live cluster map (graph + electrical-panel views) coloured by health |
| **Seguridad** | Security posture findings by severity |

The console is the human-in-the-loop control point: notifications (Teams) only alert and deep-link here; the actual decision (approve/reject/investigate) happens in the authenticated console, not in anonymous notification links.

### 14.3 Conversational Investigation — "Chat with the Cluster"

The `ClusterChatAgent` exposes the hybrid ReAct engine as a chat. The operator asks in natural language ("¿por qué falla el pod X?"); the system investigates live (streamed via SSE) and the fine-tuned ORPO expert synthesizes the final diagnosis from the gathered evidence. Safety is structural: every action passes through the read-only `kubectl_toolbox` (only `describe`/`get`/`logs`/`top`; write verbs rejected before execution).

**Deterministic scaffolding for a 1.5B model.** A free-form ReAct loop fails on a small base model: in live tests it dithered (5 consecutive THOUGHT turns with no ACTION), never drilled into the failing pod, and — most damaging — the synthesis hallucinated ("all pods are Running") because it only received the first 600 characters of `kubectl get pods -A`, where the broken pods were below the cut. The redesign moves the critical path out of the model and into the harness:

1. **Deterministic triage.** The harness always runs `kubectl get pods -A` as step 1 and extracts problem pods (CrashLoopBackOff/Error/not-ready; Completed/Succeeded and stale restarts excluded to avoid false positives), producing a compact high-signal digest.
2. **Deterministic drill-down.** The harness runs `describe` + `logs --tail` on the most severe pod, guaranteeing real root-cause evidence regardless of what the weak model emits.
3. **Scoped questions.** If the question names a real namespace ("¿cuántos pods en firmas?"), a scoped `get pods -n <ns>` runs with an authoritative deterministic count, so the expert answers the actual question instead of diagnosing the worst cluster fault.
4. **Grounded synthesis.** The expert receives the problem digest and the culprit's describe/logs (never a truncated dump), with a prompt forbidding invented pods/images/errors and write-command "fixes".

The model retains an optional breadth role (drilling additional problem pods). Guards remain: placeholder commands (`<namespace>`) are rejected, and malformed `ns/name` resource references are auto-normalized. In production this took the chat from a fabricated "everything is Running" answer to correctly quoting the live OIDC-discovery 404 behind a CrashLoopBackOff and proposing a read-only-safe fix.

### 14.4 Cluster Topology — the "Electrical Panel"

`TopologyCollector` builds a graph from 5 read-only list calls (node, pod, service, endpoints, ingress) with relationships Ingress→Service→Pod→Node and per-node health. The view offers two renderings: a layered flow graph (connectivity) and an "electrical panel" where each namespace is a board and each pod a breaker coloured by health — faults light up red at a glance.

### 14.5 Security Posture Scanner

`SecurityScanner` extends the read-only investigation from operational anomalies to **security risk**. It applies ~10 deterministic, rule-based checks over the API: privileged containers, runAsRoot, hostNetwork/PID/IPC, hostPath volumes, dangerous capabilities, mutable image tags, hardcoded secrets in env, missing resource limits, cluster-admin bindings to non-system subjects, and namespaces without NetworkPolicy. Findings carry severity (critical/high/medium/low) and a concrete recommendation.

Rule-based by design (like `trivy`/`kubescape`/`kube-bench`): security detection must be deterministic, auditable, and free of LLM hallucination. The LLM's role is complementary and *post*-detection — contextualizing and prioritizing findings (distinguishing a legitimately-privileged CNI pod from a suspicious application pod), reachable by asking the chat about a specific finding. On the production cluster the scanner surfaced 353 findings (31 critical) in under a second, with no infrastructure installed.

---

## 15. Artifacts

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
| Auto-Remediation | `src/remediation/` — risk_scorer, circuit_breaker, executor, auto_remediation, incident_store |
| Notification (pluggable) | `src/remediation/` — base_notifier, teams_notifier (Teams), notifier (email) |
| Log detection source | `src/collector/log_collector.py` (read-only application logs) |
| Topology | `src/collector/topology_collector.py` |
| Cluster chat | `src/diagnostics/cluster_chat.py` |
| Security scanner | `src/security/scanner.py` |
| Web console (5 views) | `web/server.py` · `web/static/{index,incidents,chat,topology,security}.html` |
| Evaluation results | `eval/results/eval_20260609_103514.json` (ORPO+grammar vs Hybrid+grammar) |

---

## 16. Continual Learning — Closed-Loop Fine-Tuning vs RAG

The system implements two complementary mechanisms so the RCA expert improves
from real operation, plus an empirical comparison between them. This realises the
"L5+ Learning from outcomes" previously listed as future work.

### 16.1 The learning signal is free

Every incident already carries a supervision signal: the human decision
(`response` = approved/rejected), the verification outcome (`status` = resolved/
failed), and optionally an explicit human correction. The mapping is direct:
approved+resolved → positive (chosen), rejected/failed → negative (rejected).

The blocker was that `IncidentStore` was in-memory only. Fixed with an
append-only JSONL log (`src/remediation/incident_log.py`) hooked into every
state transition, which both survives restarts and feeds the learning loop.

### 16.2 Approach A — Closed-loop preference fine-tuning (ORPO)

Pipeline (offline, GPU batch): incident outcomes → `data/feedback/feedback.jsonl`
(`dataset/feedback_capture.py`) → preference pairs `chosen`/`rejected`
(`finetune/build_loop_dataset.py`, with **experience replay** of the base dataset
to prevent catastrophic forgetting) → continual ORPO training
(`finetune/loop_train.py`, reusing `train_orpo.py`, triggered by an N-new-examples
threshold) → **non-regression gate** (`eval/gate.py`: promote only if
parse_rate/keyword_hit do not regress on a blind test set) → versioned deploy with
rollback (`finetune/deploy_model.py`: `k8s-rca-orpo-v{N}` + stable alias) →
per-version metrics in MLflow (`log_loop_cycle` → improvement curve).

Conceptually this is **continual preference fine-tuning with human feedback**
(RLHF/RLAIF family, offline, no RL), with safeguards against degenerative loops
(only verified positives / human corrections become `chosen`; gate + canary),
catastrophic forgetting (replay + non-regression), and data poisoning
(human-in-the-loop review). It changes the weights; requires a GPU; learns in
cycles.

### 16.3 Approach B — Retrieval-Augmented Generation (RAG)

`src/diagnostics/incident_retriever.py` indexes resolved/validated incidents
(TF-IDF over event text — sklearn, zero new dependencies, CPU-only) and retrieves
the most similar past cases for a new incident, injecting them (bounded to fit
`num_ctx=2048`) into the RCA prompt. The model improves by **context, not
weights**: new knowledge is available instantly, with no GPU and no forgetting.
Enabled via `RAG_ENABLED` and wired into `HybridReActAgent`.

### 16.4 Empirical comparison (RAG vs plain, same model, same test set)

`eval/compare_learning.py`, run live on CPU over the held-out test set, RAG corpus
= 1960 past cases:

| Model | parse_rate plain → RAG | keyword_hit plain → RAG |
|-------|------------------------|-------------------------|
| ORPO fine-tuned (`k8s-rca-orpo`) | 1.000 → 1.000 | 1.000 → 0.917 |
| Base (`qwen2.5:1.5b`) | 1.000 → 1.000 | 1.000 → 0.833 |

**Finding (honest):** on this domain, RAG provides **no improvement and a small
regression**. Two compounding reasons: (1) the test scenarios are keyword-detectable
from the events themselves, so both models are already at ceiling (no headroom);
(2) with a 1.5B model and a 2048-token budget, the retrieved context **competes for
and dilutes** the prompt and can distract an already-capable model. RAG's benefit
grows precisely where these don't hold: a non-specialised base model, harder cases,
or a larger context window.

### 16.5 The closed cycle — RAG as fast memory, fine-tuning as consolidation

The two approaches are wired into a single loop modelled on *complementary
learning systems* (fast hippocampal vs slow cortical memory):

1. An incident occurs; the operator investigates via chat and reaches a solution
   (recorded as a human correction — the highest-quality signal).
2. The solution lands in `feedback.jsonl`, which is the RAG index → **available
   instantly** to the next similar incident, no training.
3. When the un-consolidated feedback reaches the threshold Z, `loop_train.py`
   retrains (ORPO), the gate validates, and a promoted version is deployed (canary).
4. On promotion the registry records a **consolidation watermark** (`feedback_count`);
   the RAG retriever excludes everything up to it (`skip_consolidated`) — that
   knowledge now lives in the weights, so RAG **"empties"** of what is learned and
   keeps only newer, not-yet-consolidated cases.
5. On rollback the watermark follows the active version, so RAG **automatically
   recovers** the de-consolidated cases. No knowledge is ever lost.

This makes RAG a bounded, self-clearing working memory rather than an
ever-growing index, and fine-tuning the durable consolidation — each reinforcing
the other.

### 16.6 Decision framework

| Dimension | Fine-tuning (ORPO loop) | RAG |
|-----------|-------------------------|-----|
| GPU | Required (batch) | Not required (CPU) |
| Learning latency | Per cycle | Instant |
| Catastrophic forgetting | Risk (mitigated by replay) | None |
| Degenerative loop | Risk (mitigated by gate+canary) | None (weights untouched) |
| Explainability | Low (black box) | High (shows the cases used) |
| Format/domain adherence | Strong (what reached parse 98.6%) | Depends on base model |
| Context cost | None at inference | High (critical at num_ctx=2048) |

**Conclusion:** in *this* system fine-tuning (ORPO) is the primary mechanism — it
is what lifted RCA quality to the strong baseline measured here — while RAG is a
complementary, GPU-free continual-memory path whose marginal value is currently
limited by the tiny context window. The recommended architecture is **hybrid**:
RAG for instant, auditable day-to-day retention; periodic ORPO consolidation when
the accumulated feedback warrants it (and a GPU is available).

---

## 17. Detection & RCA Quality Hardening

Running the system continuously against a live multi-tenant cluster (≈15 namespaces, 200–600 log events per 60 s window) surfaced quality problems that synthetic benchmarks never exposed: an "anomaly flood" where almost every window scored 1.000, diagnoses that were generic or drifted into tutorial-style prose, and duplicate incidents for the same recurring problem. This section documents the fixes, each validated live.

### 17.1 Per-namespace scoring — the unit of analysis is `(namespace, window)`, not `window`

**Problem.** The detector scored the *whole window* — the template distribution of the entire cluster aggregated together. In a busy cluster any single window mixes ~15 namespaces and 30–60 templates, so the aggregate always looked "unusual" and nearly every window was flagged. A healthy namespace dragged the whole window into anomaly, and the resulting alert listed all ~15 namespaces with no clear culprit.

**Fix.** `WindowData` now tracks `ns_cluster_counts: dict[str, dict[int,int]]` — the template distribution *per namespace*. The three detection signals are computed per namespace (`severity_by_namespace`, `novelty_by_namespace`, and an Isolation Forest scored on per-namespace vectors). The window score is the **maximum over namespaces**, and the namespace achieving it is recorded as `ScoredWindow.culprit_namespace`. A healthy namespace can no longer drag the window, and every alert/incident is attributed to a single culprit. A single-namespace window reduces exactly to the old behaviour (backward compatible).

### 17.2 Absolute Isolation Forest score — the real cause of the flood

**Problem.** The IF anomaly score was normalised **min-max within the current batch** (history + new window). That makes the score *relative*: the least-normal sample of the moment is always mapped to ≈1.0, even when everything is perfectly normal. This — not the model — was the dominant driver of the flood.

**Fix.** The score is now **absolute**, derived from `IsolationForest.decision_function` (calibrated by `contamination`): `d ≥ 0` (inside the normal manifold) → 0; only the anomalous side scales, against a reference fixed at training time. A normal window now scores low and *varied* (observed 0.0–0.51) instead of saturating to 1.000. This is the single change that turned "every window is an anomaly" into "only genuinely anomalous namespaces fire".

### 17.3 Novelty warm-up after every (re)start

**Problem.** After a cold start the Drain3 parser and the IF model begin empty, so *every* template is "new" and the novelty signal saturates for a while, flooding alerts on startup.

**Fix.** A linear warm-up ramp (`NOVELTY_WARMUP_WINDOWS`, default 20) damps **only** the novelty signal for the first windows after bootstrap; severity (real errors) and IF are untouched, so genuine problems still fire immediately. Novelty is reintroduced gradually as the baseline matures.

### 17.4 RCA evidence — error-template clustering

**Problem.** The SLM received up to 40 near-identical raw error lines, which diluted the signal and consumed the `num_ctx=2048` budget, yielding generic diagnoses or "no determinable cause".

**Fix.** Error logs are grouped by `(namespace, Drain3 template)`, counted, and ranked by frequency, sending **one line per pattern plus a real example** instead of dozens of duplicates:

```
12× [postgresql] FATAL: role "<*>" does not exist
     e.g. FATAL: role "$(POSTGRES_USER)" does not exist
```

The evidence is further filtered to the **culprit namespace** so a single root cause is presented, not a cluster-wide mix.

### 17.5 Anti-drift guardrails and deterministic fallback

**Problem.** The 1.5B expert occasionally drifted into tutorial mode ("Here are some additional steps… Step 4:") or apologised ("Lo siento, parece que falta información"), producing unusable diagnoses.

**Fix.** Drift/apology markers are detected and, when the model output is unusable, the root cause is **synthesised deterministically from the dominant error template** — e.g. *"El namespace «postgresql» acumula 11 errores recurrentes del tipo: «FATAL: role <\*> does not exist»."* The system therefore never shows "no determinable cause" when real errors exist; diagnosis quality becomes independent of model variance. `stop` sequences were also added to the single-shot path.

### 17.6 Incident deduplication

**Problem.** A persistent problem (same namespaces) created one incident per detection window, flooding the incident list.

**Fix.** Incidents are deduplicated by a namespace fingerprint within a sliding window (`REMEDIATION_DEDUP_WINDOW`, default 1800 s): a recurrence increments `occurrence_count` ("visto N veces") instead of creating a new incident.

### 17.7 Remediation command quality — deterministic command builder

**Problem.** The 1.5B expert proposed `kubectl` commands that were frequently unusable: wrong namespace (`describe pod postgresql-… -n aiops-demo` when the pod lived in `postgresql`), fragile shell substitutions (`logs … $(kubectl get pod -l …)`), placeholders, or commands irrelevant to the root cause (`describe networkpolicy` for a missing-role error). On the held-out test set the fine-tuned model reached only **NS-ok 33.0%** (command names the correct namespace) and **Verb-ok 41.0%** (command verb matches the scenario).

**Fix.** A deterministic command builder (`src/diagnostics/command_builder.py`) applies the same principle used for the root cause — guarantee quality by post-processing instead of trusting model variance:

1. **Resource extraction** from the evidence: pod (`Pod/<name>`), node, PVC, service, and the owning workload (deployment/statefulset, by stripping the replicaset/pod hash suffixes).
2. **Intent → command catalog** mapping the failure pattern to the correct investigative command, with the verb aligned to each scenario (OOM/probe/image → `describe pod`; PVC → `describe pvc`; node pressure → `describe node` *without* `-n`; NetworkPolicy → `get networkpolicy`; secret/role → `get secret`; CrashLoop/config → `logs … --previous`; endpoints → `get endpoints`).
3. **Namespace correction**: the `-n` flag is forced to the detector's culprit namespace (and stripped for cluster-scoped resources like nodes).
4. **Fragility rejection**: commands with `$(...)`, pipes, `;`, or `<placeholders>` are discarded in favour of the deterministic one. The model's command is kept only when it is safe *and* consistent with the detected intent.
5. **Two-tier output**: a safe investigation command (always) plus an optional reversible remediation (`rollout restart`) tagged by the existing risk taxonomy and gated in shadow mode.

**Result (deterministic evaluation on the 210-sample test set, no model inference):**

| Metric | Fine-tuned SLM | Command builder | Δ |
|---|---|---|---|
| NS-ok% (correct namespace) | 33.0% | **85.7%** | +52.7 |
| Verb-ok% (correct verb) | 41.0% | **92.9%** | +51.9 |

The remaining ≈14% of NS-ok is the ceiling, not a miss: node-pressure scenarios resolve to `describe node`, which is cluster-scoped and *correctly* carries no namespace. As with the root-cause fallback, command quality is now independent of model variance.

### 17.8 Result

Live validation against the production cluster: normal windows now score low and varied; the only firing alerts are the cluster's *real* persistent issue (PostgreSQL `role "$(POSTGRES_USER)" does not exist`, severity = 1.00), correctly attributed to the `postgresql` namespace; benign high-volume namespaces (e.g. `argocd`, `aeat-retenciones`) no longer fire. The dashboard alert stream and the deduplicated incident list now tell the same story.

## 18. Pending / Future Work

- [x] Formal evaluation on held-out test set — implemented in `eval/run_eval.py` (210 samples, seed=99)
- [x] Alignment experiments — DPO v1/v2, SimPO, ORPO, KTO (10 experiments total)
- [x] Grammar-constrained decoding — GBNF grammar via Ollama `/api/generate`
- [x] Hybrid ReAct Agent — role separation resolves Parse%/Keyword% trade-off
- [x] Auto-remediation with human-in-the-loop — Level 1 autonomous, Level 2 approval, circuit breaker
- [x] Pluggable ChatOps notification — Microsoft Teams (primary) + email (fallback)
- [x] Web operational console — Dashboard, Incidencias, Chat, Topología, Seguridad
- [x] Dual detection source — control-plane events + application logs (read-only)
- [x] Conversational read-only investigation agent ("chat with the cluster")
- [x] Live cluster topology (graph + electrical-panel views)
- [x] Rule-based security posture scanner (10 checks, read-only)
- [x] Per-namespace anomaly scoring with single-culprit attribution (§17.1)
- [x] Absolute Isolation Forest score — eliminates the relative-normalisation anomaly flood (§17.2)
- [x] Novelty warm-up on (re)start (§17.3)
- [x] RCA evidence hardening — error-template clustering, culprit focus, anti-drift fallback (§17.4–17.5)
- [x] Incident deduplication with occurrence counter (§17.6)
- [x] Remediation command quality — deterministic command builder: NS-ok 33%→85.7%, Verb-ok 41%→92.9% (§17.7)
- [ ] Integration test: full pipeline with chaos injection on live cluster and MTTR measurement
- [ ] Shadow-mode remediation on production cluster (generate incidents, all actions gated on approval)
- [ ] LLM-assisted prioritization of security findings (rules detect, model contextualizes)
- [ ] Fine-tune on larger dataset (5k+ samples with structural diversity) — prerequisite for DPO on stronger base
- [ ] Scale to 7B model (Mistral-7B or Llama-3.1-8B) with same ORPO recipe — expected to improve both metrics
- [ ] Benchmark against GPT-4o / Claude Sonnet on same 210-sample test set
- [ ] Fine-tune investigator (Phase 1) on ReAct-format examples to improve `crash_config` scenario
- [ ] Auto-remediation: extend hybrid agent to execute safe kubectl write commands with dry-run preview
