# K8s-AIOps: Autonomous Anomaly Detection and Root Cause Analysis in Kubernetes using a Fine-Tuned Small Language Model

**Status:** Work in progress
**Model:** [aaranda233/k8s-rca-slm](https://huggingface.co/aaranda233/k8s-rca-slm)
**Hardware:** NVIDIA A30 (24GB VRAM)

---

## Abstract

This work presents an end-to-end AIOps pipeline for Kubernetes environments that combines unsupervised anomaly detection with a fine-tuned Small Language Model (SLM) for automated root cause analysis (RCA). The system collects raw Kubernetes events continuously, detects anomalous time windows using Isolation Forest, and routes flagged windows to a domain-specific SLM that produces a natural-language root cause diagnosis and a concrete `kubectl` remediation command. The SLM is derived from `Qwen2.5-1.5B-Instruct` fine-tuned via QLoRA on a curated dataset of ~986 Kubernetes incident scenarios covering 14 failure categories. The final model runs entirely on CPU at inference time, enabling deployment without GPU infrastructure.

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
│  Layer 3 — Diagnostics  (src/diagnostics/ollama_rca.py) │
│  • Formats events as plain text                         │
│  • Calls k8s-rca-slm via Ollama API                    │
│  • Parses ROOT CAUSE + KUBECTL from response            │
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

Inference parameters: `temperature=0.1`, `top_p=0.9`, `repeat_penalty=1.1`, `num_ctx=1024`.

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

## 8. Reproducibility

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

## 9. Artifacts

| Artifact | Location |
|----------|----------|
| Source code | This repository |
| Dataset (combined.jsonl) | `dataset/output/` (gitignored — regenerable) |
| LoRA adapters | `finetune/output/k8s-rca-slm/` + [HF Hub](https://huggingface.co/aaranda233/k8s-rca-slm) |
| GGUF Q8_0 (production) | [HF Hub — k8s-rca-slm-Q8_0.gguf](https://huggingface.co/aaranda233/k8s-rca-slm) |
| GGUF Q4_K_M (compact) | [HF Hub — k8s-rca-slm-Q4_K_M.gguf](https://huggingface.co/aaranda233/k8s-rca-slm) |

---

## 10. Pending / Future Work

- [ ] Formal evaluation on held-out test set (precision/recall per scenario category)
- [ ] Perplexity measurement of base vs. fine-tuned model on test set
- [ ] Integration test: full pipeline against live cluster with real anomaly injection
- [ ] Update `OLLAMA_MODEL=k8s-rca-slm` in pipeline `.env` for production use
- [ ] Investigate flash-attention 2 support (xformers fallback used during training)
- [ ] Explore training with 6+ epochs and larger dataset (2,000+ samples)
- [ ] Compare against GPT-4o / Claude Sonnet on same test set for benchmark
- [ ] Latency measurement: time-to-diagnosis from anomaly detection to RCA output
