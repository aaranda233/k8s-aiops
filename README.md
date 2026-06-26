# K8s-AIOps

Pipeline AIOps autónomo para Kubernetes: **detección** de anomalías con Isolation Forest → **diagnóstico** de causa raíz con un experto SLM fine-tuneado (+ grammar) → **remediación** con un grafo de planes ejecutables y human-in-the-loop. El diagnóstico corre en CPU sin GPU; un planner agéntico (`qwen2.5-coder:14b`, on-demand en GPU) rellena la cola larga de problemas novedosos. **Sin pasos manuales: cada acción de un plan es un comando ejecutable.**

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
2. **Diagnóstico** — el experto fine-tuneado (`k8s-rca-orpo`) produce `ROOT CAUSE` + `KUBECTL` con formato garantizado por GBNF grammar, enriquecido con un *digest* determinista de la evidencia (config "expert-only", CPU-viable; el modo híbrido sigue disponible)
3. **Remediación** — un **grafo de planes ejecutables** mapea cada firma de problema a una secuencia investigar→arreglar→verificar. Cada paso se clasifica por riesgo (L0-3) y se ejecuta con dry-run + aprobación por paso (shadow mode); los destructivos nunca se ejecutan. Ante un problema novedoso o sin acción en el catálogo, **escala al planner agéntico** (`qwen2.5-coder:14b`) que investiga en vivo y emite el comando concreto, rellenando el grafo. Circuit breaker previene bucles.

### Consola operativa (5 vistas, read-only sobre la API salvo la acción aprobada)

| Vista | Qué hace |
|-------|----------|
| **Dashboard** | El algoritmo en vivo: templates Drain3, scatter PCA, ventanas puntuadas |
| **Incidencias** | Bandeja de acciones: diagnóstico + plan multi-paso, ejecución paso a paso con botón play |
| **Topología** | Mapa del cluster en vivo (grafo + cuadro eléctrico) coloreado por salud |
| **Seguridad** | Escáner de postura: ~10 checks por severidad |
| **Grafo** | Explora el grafo de remediación: firmas, planes, origen (catálogo vs escalado agéntico) y verificación |

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

## Endurecimiento en producción

Ejecutar el sistema en continuo sobre un clúster real (~15 namespaces) destapó fallos que los benchmarks sintéticos no mostraban. Todos resueltos por **post-proceso determinista** (calidad independiente de la varianza del modelo) y validados en vivo:

- **Detección por namespace + score IF absoluto** — se puntúa cada `(namespace, ventana)` por separado (culpable = argmax) y el Isolation Forest usa `decision_function` (referencia absoluta) en vez de min-max relativo. Esto eliminó el *flood* de anomalías: las ventanas normales puntúan bajo y solo dispara el servicio que realmente falla.
- **Comandos kubectl dirigidos** — un constructor determinista extrae el recurso de la evidencia, fuerza el namespace correcto, mapea la causa al comando adecuado (catálogo intención→comando) y descarta comandos frágiles:

  | Métrica del comando | SLM (SFT) | Constructor determinista |
  |---|:---:|:---:|
  | **NS-ok%** (namespace correcto) | 33.0% | **85.7%** |
  | **Verb-ok%** (verbo correcto) | 41.0% | **92.9%** |

  *(el ≈14% restante de NS-ok es el techo: los escenarios de nodo usan `describe node`, cluster-scoped, que correctamente no lleva namespace).*

  Además, cada comando lleva una **explicación en lenguaje natural** de qué hace y qué mirar (determinista, sin modelo): `kubectl get secret -n postgresql` → *"Lista los secrets en postgresql para comprobar si falta el secret con las credenciales que el pod no encuentra"*.

- **Clasificación App / Plataforma** — cada incidente se etiqueta por dueño: **App** (config/credenciales/salud de la app) o **Plataforma** (nodo/recursos/almacenamiento/red/imagen), con badge y filtro en la bandeja. No se separan los flujos: se mantiene un solo store para preservar la **correlación eventos+logs** y la deduplicación.

- **Evidencia al SLM por plantillas, anti-deriva con fallback determinista, warm-up de novedad y deduplicación de incidentes.** Detalle en [RESEARCH.md §17](./RESEARCH.md).

---

## Documentación

| Documento | Descripción |
|-----------|-------------|
| [RESEARCH.md](./RESEARCH.md) | Paper principal — arquitectura, 10 experimentos, agente híbrido, grafo de remediación + planner agéntico, auto-remediación |
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
│   ├── diagnostics/        # Capa 3: single_shot · react · hybrid + grammar + escalation + agentic_planner + toolbox
│   ├── remediation/        # Capa 4: remediation_graph · risk_scorer · circuit_breaker · executor · notifier · incidentes
│   ├── security/           # Escáner de postura de seguridad (read-only)
│   └── tracking/           # MLflow
├── tests/                  # 414 tests (pytest)
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

El RBAC es **read-only por defecto**. Los permisos de escritura para el conjunto de acciones reversibles/config (`rollout restart`/`undo`, `scale`, `set image`/`resources`/`env`) solo se conceden con `rbac.allowRemediation=true`; `delete`/`drain`/`exec`/`apply`/`create` nunca.

### Evaluación

```bash
python eval/run_eval.py --models orpo,hybrid_orpo --host http://<ollama-host>:11434
```

### Tests

```bash
pytest tests/ -v        # 414 tests — riesgo, circuit breaker, executor, grafo, escalado agéntico, web
```

---

## Modos de diagnóstico (`REACT_MODE`)

| Modo | Descripción | Cuándo usar |
|------|-------------|-------------|
| `single_shot` | Una llamada al experto + grammar + digest determinista | **Producción** — CPU-viable (~0.7s GPU / ~32s CPU), Keyword% 83% |
| `react` | Loop ReAct con el fine-tuneado | Experimental (1.5B no sigue bien el formato ReAct) |
| `hybrid` | Base investiga + experto diagnostica + grammar | Mejor Keyword% (~93%) si hay presupuesto de latencia GPU |

---

## Hardware

| Componente | Especificación |
|-----------|----------------|
| Entrenamiento | NVIDIA A30 24GB · unsloth + TRL |
| Inferencia | Intel Xeon Gold 6526Y · Ollama · GGUF Q8_0 — **sin GPU** |
| Cluster | Kubernetes |

---

## Estado del proyecto

**Completado (versión final):** pipeline 4 capas · 10 experimentos de alineación · experto single-shot + grammar + digest (CPU-viable) · agente híbrido · **grafo de remediación ejecutable** · **planner agéntico** (`qwen2.5-coder:14b`, on-demand) que rellena la cola larga · **sin pasos manuales** (toda acción es un comando ejecutable; lo externo se marca como nota) · vista Grafo + ejecución paso a paso · auto-remediación con human-in-the-loop · 414 tests · CI/CD · Docker · Helm.

**Próximo:** test de integración con chaos injection en cluster real + medición de MTTR · benchmark vs GPT-4o/Claude · consolidación periódica del grafo verificado a ORPO.
