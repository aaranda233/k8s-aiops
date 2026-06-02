# K8s-AIOps

Pipeline AIOps para Kubernetes: detección de anomalías con Isolation Forest + diagnóstico de causa raíz con un SLM fine-tuneado sobre datos del cluster.

**Modelo:** [aaranda233/k8s-rca-slm](https://huggingface.co/aaranda233/k8s-rca-slm) · **Dataset:** [aaranda233/k8s-rca-dataset](https://huggingface.co/datasets/aaranda233/k8s-rca-dataset) · **Informe:** [GitHub Pages](https://aaranda233.github.io/k8s-aiops/)

---

## Documentación del proyecto

| Documento | Descripción | Fecha |
|-----------|-------------|-------|
| [RESEARCH.md](./RESEARCH.md) | Paper principal — arquitectura completa del pipeline, motivación, abstract, resultados de inferencia CPU | May 2026 |
| [RESEARCH_DETECTION.md](./RESEARCH_DETECTION.md) | Detalle técnico de las capas 1 y 2: Watch API, Drain3, ventanas temporales, Isolation Forest con reentrenamiento continuo | May 2026 |
| [eval/EVAL_RESULTS.md](./eval/EVAL_RESULTS.md) | Resultados de evaluación cuantitativa: SFT vs baseline vanilla en 210 muestras ciegas — tabla comparativa con parse rate, ROUGE-L, keyword hit, latencia | Jun 2026 |

---

## Estructura del repositorio

```
k8s-aiops/
│
├── src/                        # Pipeline principal
│   ├── collector/              # Capa 1: Watch API Kubernetes
│   ├── parser/                 # Capa 1: Drain3 log parsing
│   ├── detector/               # Capa 2: Isolation Forest + ventanas
│   ├── diagnostics/            # Capa 3: OllamaRCA (SLM)
│   └── tracking/               # MLflow tracker
│
├── dataset/                    # Generación del dataset de entrenamiento
│   ├── scenarios/              # YAMLs con los 14 escenarios de fallo
│   ├── generator.py            # Genera variaciones sintéticas (seed=42)
│   └── output/combined.jsonl  # 986 muestras en formato ChatML
│
├── finetune/                   # Fine-tuning QLoRA
│   ├── train.py                # Entrenamiento con unsloth + SFTTrainer
│   ├── Modelfile               # Configuración Ollama (Q4_K_M)
│   └── log_finetune_to_mlflow.py  # Sube métricas retroactivas a MLflow
│
├── eval/                       # Harness de evaluación
│   ├── metrics.py              # ROUGE-L, keyword oracle, parse rate
│   ├── runner.py               # Inferencia multi-modelo sobre Ollama
│   ├── run_eval.py             # Orquestador — genera tabla comparativa
│   ├── test_set.jsonl          # 210 muestras ciegas (seed=99)
│   ├── results/                # JSONs con resultados por ejecución
│   └── EVAL_RESULTS.md         # Resultados y análisis (Jun 2026)
│
├── web/                        # Dashboard web tiempo real
│   ├── server.py               # FastAPI + WebSocket
│   └── static/index.html       # Dashboard (D3.js + Chart.js)
│
├── config/                     # Configuración del pipeline
│   └── settings.py             # PipelineConfig, MLflowConfig, etc.
│
├── report/                     # Informe para tesis
│   ├── index.html              # Informe técnico completo (tema claro)
│   └── dashboard-demo.html     # Mock animado del dashboard
│
├── docs/                       # GitHub Pages (protegido con contraseña)
│   ├── index.html              # Wrapper con SHA-256 (pw: k8s2026)
│   └── dashboard-demo.html     # Demo embebida en el informe
│
├── main.py                     # Punto de entrada (--replay / --live)
└── requirements.txt
```

---

## Progreso del proyecto

### Completado

| Hito | Fecha |
|------|-------|
| Pipeline completo: collector → parser → detector → RCA | Abr 2026 |
| Dataset sintético: 14 escenarios × 70 muestras = 986 samples | Abr 2026 |
| Fine-tuning QLoRA sobre Qwen2.5-1.5B-Instruct (A30 24GB) | May 2026 |
| Modelo en HuggingFace + GGUF Q4_K_M para CPU | May 2026 |
| Validación inferencia CPU (Intel Xeon Gold 6526Y): ~0.83s/resp | May 2026 |
| Dashboard web con WebSocket (D3.js scatter PCA + Chart.js) | May 2026 |
| MLflow tracking integrado (experimentos: k8s-aiops, k8s-aiops-finetune) | May 2026 |
| Dataset subido a HuggingFace Hub | May 2026 |
| Informe técnico web publicado en GitHub Pages | May 2026 |
| **Harness de evaluación: 210 muestras ciegas, SFT vs baseline** | Jun 2026 |

### En curso / Próximos pasos

| Paso | Prioridad | Objetivo |
|------|-----------|----------|
| Structured outputs (grammar sampling en Ollama) | Alta | Parse% → ~100% sin reentrenar |
| Ablación Q4_K_M vs Q8_0 vs fp16 con el harness | Media | Tabla eficiencia para paper |
| DPO fine-tuning con pares chosen/rejected | Media | Keyword% ≥ 80%, reducir alucinación |

---

## Resultados clave (evaluación Jun 2026)

| Métrica | SFT (nuestro) | Baseline vanilla |
|---------|:---:|:---:|
| Sigue el formato ROOT CAUSE/KUBECTL | **56.2%** | 38.6% |
| Keywords del fallo correctas | 60.0% | **92.4%** |
| ROUGE-L vs referencia | **56.7%** | 2.5% |
| Namespace correcto en kubectl | **33.0%** | 1.4% |
| Verbo kubectl correcto | **41.0%** | 1.9% |
| Latencia media (CPU) | **0.83s** | 0.96s |

El SFT mejora el formato y la estructura del kubectl pero pierde generalización (sobreajuste en 986 muestras). El baseline conoce más vocabulario técnico pero no estructura la respuesta. Próximo paso: structured outputs + DPO.

---

## Arrancar el pipeline

```bash
# Instalar dependencias
pip install -r requirements.txt

# Modo replay (eventos históricos del cluster)
python main.py --replay

# Modo live (Watch API en tiempo real)
python main.py --live

# Dashboard web
# http://localhost:8765 (arranca automáticamente con --live)

# Evaluación del modelo
python eval/run_eval.py --samples 15 --host http://<ollama-host>:11434
```

---

## Hardware utilizado

| Componente | Especificación |
|-----------|----------------|
| Entrenamiento | NVIDIA A30 24GB VRAM · unsloth + trl SFTTrainer |
| Inferencia | Intel Xeon Gold 6526Y · Ollama · GGUF Q4_K_M |
| Cluster | Kubernetes (microkube) · 192.168.2.204 |
| MLflow | NodePort 30803 · http://192.168.2.204:30803 |
