# Evaluación del SLM K8s-RCA — Resultados

**Fecha:** 2026-06-02 (SFT+Baseline) / 2026-06-02 (DPO — negativo) / 2026-06-02 (SimPO — negativo)
**Test set:** 210 muestras ciegas (seed=99 ≠ seed entrenamiento=42)
**Escenarios:** 14 × 15 muestras/escenario
**Modelos:** `k8s-rca-slm` (SFT QLoRA) · `k8s-rca-dpo` (DPO) · `k8s-rca-simpo` (SimPO) · `qwen2.5:1.5b` (baseline vanilla)
**Infraestructura:** CPU Intel Xeon Gold 6526Y · Ollama · GGUF Q4_K_M/Q8_0

---

## Tabla comparativa (4 modelos)

| Métrica | SFT (nuestro) | DPO | SimPO | Baseline (vanilla) |
|---------|:---:|:---:|:---:|:---:|
| **Parse%** — sigue formato ROOT CAUSE/KUBECTL | 56.2% | 0.0% | 0.5% | 38.6% |
| **Keyword%** — menciona las palabras clave del fallo | 60.0% | 0.0% | 0.0% | **92.4%** |
| **ROUGE-L** — similitud con respuesta de referencia | **56.7%** | 0.0% | 0.0% | 2.5% |
| **NS-ok%** — kubectl incluye el namespace correcto | **33.0%** | 0.0% | 0.0% | 1.4% |
| **Verb-ok%** — kubectl usa el verbo correcto (logs/describe/get) | **41.0%** | 28.6% | 28.1% | 1.9% |
| **Latencia media** | **0.82s** | 2.04s | 1.60s | 0.98s |
| **Latencia p95** | **1.24s** | 2.25s | 2.21s | 1.38s |

*N = 210 muestras por modelo · seed=99*

> **DPO y SimPO = resultados negativos**: ambos modelos colapsaron formato y vocabulario. El Verb-ok% ~28% en los dos modelos fallidos es texto en prosa que casualmente contiene verbos de kubectl, no comandos reales. Ver análisis en sección [Diagnóstico de colapso por preference optimization](#diagnóstico-de-colapso-por-preference-optimization).

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

## Diagnóstico de colapso por preference optimization

### Patrón común — DPO y SimPO

Ambos experimentos producen el mismo colapso con mecanismos distintos:

| | DPO v1 | SimPO |
|---|---|---|
| β | 0.05 | 2.0 |
| γ | — | 1.0 |
| cpo_alpha | — | 0.1 |
| Épocas | 2 | 1 |
| Loss paso 1 | ≈ 0 (señal nula) | 0.045 (señal real) |
| Loss final | 3.04×10⁻⁷ | 0.011 |
| rewards/margins final | — | 12.47 |
| Parse% | 0.0% | 0.5% |
| Keyword% | 0.0% | 0.0% |

**DPO v1:** π_θ ≈ π_ref desde el inicio → gradiente ≈ 0 → corrupciones aleatorias acumuladas.

**SimPO:** señal real pero β=2.0 demasiado alto → el modelo aprende a reducir log π(rejected) → −∞ en lugar de mejorar los chosen. Resultado: la mitad de las respuestas son casi vacías (lat ~0.2s) y la otra mitad son prosa sin estructura (lat ~2s).

### Causa raíz compartida

El SFT sobreajustó sobre 986 ejemplos sintéticos (loss=0.089). Cualquier método de preference optimization sobre un modelo memorizado converge sobre una representación frágil que colapsa ante gradientes de preferencia:

- Los pares chosen/rejected vienen de la misma distribución memorizada
- Las capas de atención que codifican el formato `ROOT CAUSE: / KUBECTL:` son sensibles a pequeñas perturbaciones
- No existe generalización que el preference learning pueda mejorar — el modelo ya es "perfecto" en su distribución de entrenamiento

### Lecciones consolidadas

| Problema | Corrección |
|----------|-----------|
| SFT sobreajustado | Más datos (≥5k) con mayor variación antes de preference optimization |
| DPO: ref_model = política entrenada | Usar modelo base vanilla como ref_model |
| SimPO: β=2.0 demasiado agresivo | β ≤ 0.5 para datasets pequeños |
| Ambos: dataset de 818 pares insuficiente | Mínimo 2k pares con mayor diversidad semántica |

---

## DPO — Diagnóstico de fallo

### Configuración del experimento

| Parámetro | Valor |
|-----------|-------|
| Base de partida | `k8s-rca-slm` (SFT QLoRA 4-bit) |
| Dataset | 541 pares chosen/rejected (de 986 muestras SFT) |
| β (beta) | 0.05 |
| Épocas | 2 |
| LR | 5e-5 |
| Batch efectivo | 16 (batch=1 × grad_accum=16) |
| Runtime | 12.1 min · A30 24GB |

### Síntomas observados

1. **Loss ≈ 0 desde el paso 1** — el modelo SFT ya asignaba probabilidad fuertemente mayor a los `chosen` que a los `rejected` antes de cualquier actualización. La señal de gradiente DPO fue prácticamente nula durante todo el entrenamiento (loss final: 3.04×10⁻⁷).
2. **Colapso total en evaluación** — Parse%=0%, Keyword%=0%, ROUGE-L=0%. El modelo generó texto libre sin estructura en todas las 210 muestras.
3. **Verb-ok%=28.6%** — única métrica no nula: casualmente menciona verbos de kubectl pero en respuestas completamente desestructuradas.

### Causa raíz

El SFT había memorizado las 986 muestras de entrenamiento con una **confianza extremadamente alta** (loss SFT final = 0.0898). Cuando el DPOTrainer calculó los logprobs de referencia, encontró que la política SFT ya distinguía perfectamente `chosen` de `rejected` — el margen de preferencia era máximo desde el inicio.

Con **β=0.05** (muy bajo), la penalización KL por alejarse de la referencia era mínima. El optimizador recibió gradientes casi nulos, pero los pocos pasos de actualización que sí ocurrieron **corrompieron los pesos críticos** del formato (capas de atención que codificaban la estructura `ROOT CAUSE: / KUBECTL:`), destruyendo el formato aprendido en el SFT.

### Lecciones

| Problema | Causa | Corrección para siguiente intento |
|----------|-------|-----------------------------------|
| Loss≈0 desde paso 1 | Modelo SFT sobreajustado = referencia demasiado fuerte | Usar modelo base vanilla como referencia en lugar del SFT memorizado |
| β=0.05 insuficiente | Regularización KL demasiado débil para señal tan pequeña | Subir β a 0.2–0.5 |
| 2 épocas destructivas | Con gradientes mínimos, múltiples épocas acumulan ruido | Reducir a 1 época; añadir early stopping por reward margin |
| Dataset rejected de baja calidad | El modelo vanilla (temp=0.7) no siempre genera respuestas claramente erróneas | Usar temperatura más baja (0.3) para el modelo vanilla al generar rejected |

---

## Próximos pasos

### 1. Structured outputs (impacto inmediato, sin reentrenar)
Forzar el formato `ROOT CAUSE: / KUBECTL:` con grammar-constrained sampling en llama.cpp/Ollama.
**Objetivo:** Parse% → ~100%
**Coste:** ninguno (configuración, no reentrenamiento)

### 2. Preference optimization v3 *(bloqueado hasta resolver causa raíz)*
DPO v1 y SimPO han confirmado que el bloqueante no es la variante de loss sino el SFT sobreajustado. El prerequisito es ampliar el dataset SFT (≥5k muestras, mayor variación de namespaces/pod names/mensajes de error) y reentrenar antes de reintentar preference optimization.

**Prerrequisito:** SFT con loss > 0.3 (generalización real, no memorización)
**Entonces:** DPO con ref_model=vanilla, β=0.2, 1 época

### 3. Ablation de cuantización (resultado de eficiencia)
Comparar Q4_K_M vs Q8_0 vs fp16 con el mismo test set.
**Resultado esperado para el paper:** "retenemos X% de precisión a 1/4 del tamaño"

---

## Artefactos

| Archivo | Descripción |
|---------|-------------|
| `eval/test_set.jsonl` | 210 muestras ciegas (seed=99) |
| `eval/results/eval_20260601_154131.json` | Resultados SFT+Baseline por muestra |
| `eval/results/eval_20260602_090116.json` | Resultados SFT+DPO+Baseline (3 modelos) |
| `eval/results/eval_20260602_094119.json` | Resultados SFT+Baseline (confirmación) |
| `eval/results/eval_20260602_*.json` | Resultados SimPO (210 muestras, colapso) |
| `eval/metrics.py` | ROUGE-L, keyword oracle, parse rate |
| `eval/runner.py` | Inferencia multi-modelo sobre Ollama |
| `eval/run_eval.py` | Orquestador principal |
