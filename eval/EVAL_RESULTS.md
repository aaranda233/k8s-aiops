# Evaluación del SLM K8s-RCA — Resultados

**Fecha:** 2026-06-02
**Test set:** 210 muestras ciegas (seed=99 ≠ seed entrenamiento=42)
**Escenarios:** 14 × 15 muestras/escenario
**Modelos:** `k8s-rca-slm` (SFT QLoRA) vs `qwen2.5:1.5b` (baseline vanilla)
**Infraestructura:** CPU Intel Xeon Gold 6526Y · Ollama · GGUF Q4_K_M

---

## Tabla comparativa

| Métrica | SFT (nuestro) | Baseline (vanilla) | Delta |
|---------|:---:|:---:|:---:|
| **Parse%** — sigue formato ROOT CAUSE/KUBECTL | 56.2% | 38.6% | +17.6 pp |
| **Keyword%** — menciona las palabras clave del fallo | 60.0% | **92.4%** | −32.4 pp |
| **ROUGE-L** — similitud con respuesta de referencia | **56.7%** | 2.5% | +54.2 pp |
| **NS-ok%** — kubectl incluye el namespace correcto | 33.0% | 1.4% | +31.6 pp |
| **Verb-ok%** — kubectl usa el verbo correcto (logs/describe/get) | 41.0% | 1.9% | +39.1 pp |
| **Latencia media** | **0.83s** | 0.96s | −0.13s |
| **Latencia p95** | **1.24s** | 1.38s | −0.14s |

*N = 210 muestras por modelo*

---

## Keyword hit por escenario (SFT vs Baseline)

| Escenario | SFT | Baseline |
|-----------|:---:|:---:|
| crash_config | — | 53.3% |
| crash_oom | — | 100.0% |
| crash_probe | — | 100.0% |
| crash_secret | — | 100.0% |
| image_auth | — | 100.0% |
| image_not_found | — | 100.0% |
| image_registry_down | — | 100.0% |
| network_policy_block | — | 66.7% |
| node_disk_pressure | — | 100.0% |
| node_pressure_memory | — | 100.0% |
| pending_insufficient_cpu | — | 100.0% |
| pvc_pending | — | 93.3% |
| readiness_failing | — | 100.0% |
| service_no_endpoints | — | 80.0% |

> El desglose SFT por escenario está disponible en `eval/results/eval_20260601_154131.json`

---

## Interpretación

### Qué aprendió el SFT
- **Formato (+17.6 pp):** el SFT sigue la estructura `ROOT CAUSE: / KUBECTL:` con mucha más consistencia que el vanilla. El baseline frecuentemente responde en prosa libre.
- **Estructura del kubectl (+31–39 pp):** incluye el namespace y el verbo correcto con una frecuencia radicalmente superior al vanilla (que casi nunca los incluye).
- **Similitud textual (+54 pp en ROUGE-L):** cuando parsea correctamente, reproduce respuestas muy cercanas a las de referencia.

### Qué perdió el SFT
- **Vocabulario técnico (−32.4 pp en keywords):** al memorizar los outputs de 986 muestras, el modelo perdió generalización. Ante variaciones del mismo escenario (namespace distinto, pod distinto), no menciona los conceptos correctos con la misma frecuencia que el modelo base.
- **Parse rate bajo (56%):** en 44% de las respuestas no sigue el formato esperado. Causa más probable: conflicto entre el system prompt en inferencia y el formato aprendido en training.

### Diagnóstico principal
> El SFT presenta **sobreajuste de formato**: aprendió a copiar la estructura y el texto exacto de las 986 muestras de entrenamiento, pero no generalizó el conocimiento subyacente. Al ver muestras con variaciones (seed diferente), pierde vocabulario técnico aunque mantiene la estructura.

Esto es consistente con un **loss final de 0.0898 sobre 986 muestras** — señal clásica de memorización.

---

## Próximos pasos

### 1. Structured outputs (impacto inmediato, sin reentrenar)
Forzar el formato `ROOT CAUSE: / KUBECTL:` con grammar-constrained sampling en llama.cpp/Ollama.
**Objetivo:** Parse% → ~100%
**Coste:** ninguno (configuración, no reentrenamiento)

### 2. DPO fine-tuning (contribución metodológica principal)
Construir pares `(chosen, rejected)` donde:
- **chosen:** respuesta correcta del dataset original
- **rejected:** salida del modelo base pre-SFT o variante alucinada

DPO como mecanismo explícito de reducción de alucinación — el keyword gap (−32 pp) es el argumento cuantificado que lo justifica.
**Objetivo:** Keyword% ≥ 80% manteniendo Parse% y ROUGE-L
**Criterio de éxito:** este mismo harness

### 3. Ablation de cuantización (resultado de eficiencia)
Comparar Q4_K_M vs Q8_0 vs fp16 con el mismo test set.
**Resultado esperado para el paper:** "retenemos X% de precisión a 1/4 del tamaño"

---

## Artefactos

| Archivo | Descripción |
|---------|-------------|
| `eval/test_set.jsonl` | 210 muestras ciegas (seed=99) |
| `eval/results/eval_20260601_154131.json` | Resultados completos por muestra |
| `eval/metrics.py` | ROUGE-L, keyword oracle, parse rate |
| `eval/runner.py` | Inferencia multi-modelo sobre Ollama |
| `eval/run_eval.py` | Orquestador principal |
