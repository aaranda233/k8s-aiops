# K8s-AIOps

Pipeline AIOps autónomo para Kubernetes: **detección** de anomalías con Isolation Forest → **diagnóstico** de causa raíz con un agente híbrido (SLM fine-tuneado + grammar) → **remediación** con human-in-the-loop. Todo corre en CPU, sin GPU en inferencia.

**Modelos:** [k8s-rca-slm](https://huggingface.co/aaranda233/k8s-rca-slm) · [k8s-rca-orpo](https://huggingface.co/aaranda233/k8s-rca-orpo) — **Dataset:** [k8s-rca-dataset](https://huggingface.co/datasets/aaranda233/k8s-rca-dataset) — **Informe:** [GitHub Pages](https://aaranda233.github.io/k8s-aiops/)

---

## Qué hace

```
┌─ Capa 1 ─────────┐   ┌─ Capa 2 ─────────┐   ┌─ Capa 3 ──────────────┐   ┌─ Capa 4 ──────────┐
│ Watch API K8s    │ → │ Isolation Forest │ → │ Agente Híbrido ReAct  │ → │ Auto-remediación  │
│ + Drain3 parsing │   │ (no supervisado) │   │ investiga + diagnostica│   │ human-in-the-loop │
└──────────────────┘   └──────────────────┘   └───────────────────────┘   └───────────────────┘
```

1. **Detección** — eventos del cluster **+ logs de aplicación** en ventanas de 60s; Isolation Forest con reentrenamiento continuo marca ventanas anómalas
2. **Diagnóstico** — un modelo base (`qwen2.5:1.5b`) investiga con kubectl de solo lectura (THOUGHT/ACTION), luego un experto fine-tuneado (`k8s-rca-orpo`) produce `ROOT CAUSE` + `KUBECTL` con formato garantizado por GBNF grammar
3. **Remediación** — clasifica el comando por riesgo (Level 0-3): Level 1 reversible se ejecuta solo + verifica, Level 2 requiere aprobación, Level 3 destructivo nunca se ejecuta. Circuit breaker previene bucles.

### Consola operativa (5 vistas, todo read-only sobre la API)

| Vista | Qué hace |
|-------|----------|
| **Dashboard** | El algoritmo en vivo: templates Drain3, scatter PCA, ventanas puntuadas |
| **Incidencias** | Bandeja de acciones: diagnóstico + kubectl propuesto, aprobar/rechazar |
| **Chat** | Investigación conversacional del cluster (ReAct read-only, streaming) |
| **Topología** | Mapa del cluster en vivo (grafo + cuadro eléctrico) coloreado por salud |
| **Seguridad** | Escáner de postura: ~10 checks por severidad |

Notificación pluggable: **Microsoft Teams** (principal) + email (fallback). Teams avisa y enlaza a la consola; la decisión humana ocurre en la web.

---

## Resultados clave (210 muestras ciegas, seed=99)

| Métrica | Baseline | SFT v1 | ORPO+grammar | **Hybrid+grammar** |
|---------|:---:|:---:|:---:|:---:|
| **Parse%** (sigue formato) | 38.6% | 56.2% | **100.0%** | 98.6% |
| **Keyword%** (acierta el fallo) | 92.4% | 60.0% | 78.1% | **92.9%** |
| **NS-ok%** (namespace correcto) | 1.4% | 33.0% | **89.5%** | 73.3% |
| Latencia media (CPU) | 1.00s | 0.86s | **0.71s** | 2.04s |

> **Hallazgo principal:** el trade-off Parse%/Keyword% que aparece en los 9 experimentos de fine-tuning no es un límite del enfoque, sino consecuencia de resolver dos objetivos con un solo modelo pequeño. La separación de roles (investigador + experto) lo resuelve: el híbrido iguala el Keyword% del baseline (92.9%) manteniendo Parse% ~100%. Detalle en [RESEARCH.md](./RESEARCH.md) y [EXPERIMENTS.md](./EXPERIMENTS.md).

---

## Documentación

| Documento | Descripción |
|-----------|-------------|
| [RESEARCH.md](./RESEARCH.md) | Paper principal — arquitectura, 10 experimentos, agente híbrido, auto-remediación |
| [RESEARCH_DETECTION.md](./RESEARCH_DETECTION.md) | Detalle de capas 1-2: Watch API, Drain3, Isolation Forest |
| [EXPERIMENTS.md](./EXPERIMENTS.md) | Registro científico de los 10 experimentos de alineación |
| [eval/EVAL_RESULTS.md](./eval/EVAL_RESULTS.md) | Resultados cuantitativos completos por modelo y escenario |

---

## Estructura

```
k8s-aiops/
├── src/
│   ├── collector/          # Capa 1: eventos (Watch API) + logs de app (read-only) + topología
│   ├── parser/             # Capa 1: Drain3
│   ├── detector/           # Capa 2: Isolation Forest + ventanas
│   ├── diagnostics/        # Capa 3: single_shot · react · hybrid + grammar + chat + toolbox
│   ├── remediation/        # Capa 4: risk_scorer · circuit_breaker · executor · notifier · incidentes
│   ├── security/           # Escáner de postura de seguridad (read-only)
│   └── tracking/           # MLflow
├── tests/                  # 119 tests (pytest)
├── eval/                   # Harness + 210 muestras ciegas + resultados
├── finetune/               # SFT · DPO · ORPO · KTO · SimPO + Modelfiles
├── dataset/                # Generador + 14 escenarios YAML
├── web/                    # Consola: server FastAPI + 5 vistas (static/*.html)
├── helm/k8s-aiops/         # Chart con RBAC de permisos mínimos
├── docs/ · report/         # Informe web (GitHub Pages)
├── Dockerfile · docker-compose.yml
├── .env.example
└── main.py
```

---

## Arranque rápido

### Local (CLI)

```bash
pip install -r requirements.txt
cp .env.example .env          # ajusta OLLAMA_HOST, OLLAMA_MODEL, etc.

python main.py replay         # procesa eventos históricos
python main.py live           # stream en vivo (Watch API)
```

### Stack completo con Docker

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:1.5b
# UI en http://localhost:8000
```

### En el cluster (Helm)

```bash
# Modo seguro: solo lectura, sin remediación
helm install aiops helm/k8s-aiops/ -n aiops --create-namespace

# Con remediación Level 1 + permisos de escritura + notificación Teams
helm install aiops helm/k8s-aiops/ -n aiops --create-namespace \
  --set remediation.enabled=true \
  --set rbac.allowRemediation=true \
  --set remediation.notifyChannel=teams \
  --set remediation.teamsWebhookUrl=https://prod.westeurope.logic.azure.com/workflows/...
```

Las aprobaciones llegan al canal de **Microsoft Teams** del equipo como Adaptive Cards con botones APROBAR/RECHAZAR. Canal configurable (`NOTIFY_CHANNEL=teams|email|both`).

El RBAC es **read-only por defecto**. Los permisos de escritura (`patch`/`scale` sobre deployments) solo se conceden con `rbac.allowRemediation=true`; `delete`/`drain` nunca.

### Evaluación

```bash
python eval/run_eval.py --models orpo,hybrid_orpo --host http://<ollama-host>:11434
```

### Tests

```bash
pytest tests/ -v        # 65 tests — riesgo, circuit breaker, executor, notifier
```

---

## Modos de diagnóstico (`REACT_MODE`)

| Modo | Descripción | Cuándo usar |
|------|-------------|-------------|
| `single_shot` | Una llamada al fine-tuneado | Más rápido (0.7s), menor cobertura |
| `react` | Loop ReAct con el fine-tuneado | Experimental (1.5B no sigue bien el formato ReAct) |
| `hybrid` | Base investiga + experto diagnostica + grammar | **Recomendado** — mejor Keyword% |

---

## Hardware

| Componente | Especificación |
|-----------|----------------|
| Entrenamiento | NVIDIA A30 24GB · unsloth + TRL |
| Inferencia | Intel Xeon Gold 6526Y · Ollama · GGUF Q8_0 — **sin GPU** |
| Cluster | Kubernetes |

---

## Estado del proyecto

**Completado:** pipeline 4 capas · 10 experimentos de alineación · agente híbrido + grammar · auto-remediación con human-in-the-loop · 65 tests · CI/CD · Docker · Helm · health checks.

**Próximo:** test de integración con chaos injection en cluster real + medición de MTTR · benchmark vs GPT-4o/Claude · escalar a modelo 7B.
