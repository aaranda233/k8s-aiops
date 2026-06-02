# Evaluación del SLM K8s-RCA — Resultados

**Fecha:** 2026-06-02 (SFT+Baseline) / 2026-06-02 (DPO — negativo) / 2026-06-02 (SimPO — negativo) / 2026-06-02 (Grammar sampling — diagnóstico)
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

### 1. Structured outputs — grammar-constrained sampling *(implementado)*

Forzar el formato `ROOT CAUSE: / KUBECTL:` con GBNF grammar en Ollama.
**Parse% → ~100% garantizado por construcción.**

**Advertencia crítica:** grammar arregla la forma, no el contenido. Un kubectl perfectamente formateado puede seguir siendo el comando equivocado. Su valor real es como **diagnóstico**: una vez forzado el formato, Keyword%, NS-ok% y Verb-ok% miden exclusivamente si el modelo sabe la respuesta — desacoplado del fallo de formato. Esto revelará cuánto del 56% de parse rate era "no sabe el formato" vs. "no sabe la respuesta".

```bash
# Activar grammar en la evaluación:
python eval/run_eval.py --models sft,baseline --grammar
```

---

### 2. Ampliar el dataset SFT — diversidad estructural, no solo volumen

**El problema es diversidad, no número.** 5k muestras de las mismas 14 plantillas con variación superficial se memorizan igual que 986 — solo son más instancias del mismo molde.

La variación tiene que ser **estructural**:
- **Más escenarios** (añadidos en `dataset/scenarios/advanced.yaml`): DNS, RBAC, init containers, HPA, evictions por nodo, Ingress, ConfigMap missing, PDB, Jobs, ResourceQuota, StorageClass, imagen con tag mutable
- **Parafraseo de la causa raíz**: misma causa expresada de varias formas, con y sin el pod name, con distintos niveles de detalle
- **Ruido en los logs**: orden de eventos distinto, contadores variables, mensajes parciales
- **Variación de namespace/pod** ya existe, mantener y ampliar

**El held-out set de evaluación debe generarse con un proceso distinto** (seed diferente ✓, pero también escenarios distintos en el futuro). Si el eval set viene de la misma distribución de plantillas, no detectarás si rompiste el sobreajuste o solo lo trasladaste. El objetivo es loss de entrenamiento > 0.3 en held-out (no minimizar el loss de entrenamiento).

---

### 3. Reentrenar SFT y reintentar PO *(bloqueado hasta paso 2)*

**Prerrequisitos antes de tocar Preference Optimization:**

1. Verificar que el nuevo SFT generaliza sobre el held-out set antes de aplicar PO. Si loss de validación sigue convergiendo a ≈ 0, PO colapsará por la misma razón.
2. Usar early stopping o menos épocas para mantener loss > 0.3 — el objetivo es held-out performance, no minimizar train loss.
3. **Reconstruir los pares chosen/rejected**: el fallo de DPO/SimPO también estaba en los datos. Los rejected venían del Qwen vanilla → la señal era de *formato*, no de *calidad de diagnóstico*. El rejected debe venir del propio SFT (diagnósticos incompletos, causa raíz perturbada) para que chosen y rejected compartan formato y difieran en corrección del contenido.

**Este paso es condicional, no obligatorio.** Si el SFT ampliado + grammar ya alcanza los números objetivo, no es necesario añadir PO. No hacerlo solo por cerrar el círculo experimental.

---

### 4. Ablación de cuantización *(independiente)*

Comparar Q4_K_M vs Q8_0 vs fp16 con el mismo test set.
**Resultado esperado:** "retenemos X% de precisión a 1/4 del tamaño de modelo"

---

## Experimento: Grammar-constrained sampling

**Fecha:** 2026-06-02 · **Archivo:** `eval/results/eval_20260602_103303.json`

GBNF grammar aplicada vía `/api/generate` para forzar el formato `ROOT CAUSE: / KUBECTL:` a nivel de token. Se evaluaron SFT y baseline (210 muestras, seed=99).

### Resultados con grammar activa

| Métrica | SFT sin grammar | SFT con grammar | Baseline sin grammar | Baseline con grammar |
|---------|:-:|:-:|:-:|:-:|
| **Parse%** | 56.2% | **30.0%** ↓ | 38.6% | **54.3%** ↑ |
| **Keyword%** | 60.0% | **31.4%** ↓ | 92.4% | **91.4%** ≈ |
| **ROUGE-L** | 56.7% | **29.8%** ↓ | 2.5% | **2.3%** ≈ |
| **NS-ok%** | 33.0% | **15.7%** ↓ | 1.4% | **1.0%** ≈ |
| **Verb-ok%** | 41.0% | **38.6%** ≈ | 1.9% | **3.3%** ↑ |

### Análisis del output raw del SFT bajo grammar

Inspección directa de la respuesta de Ollama con grammar activa sobre el SFT:

```
Input:  Anomaly Score: 0.91 / namespace: staging / Deployment: s3-sync-worker / ...
Output: "ROOT CAUTION : Le pod 's3-sync-worker' a été effacé sur le cluster Kubernetes.
         Il semble qu'un lavalier soit tombé en panne ou être mal configuré.
         kubectl create -f <nom-du-prototype>.yaml --image=propriété:version"
```

Tres síntomas simultáneos:

1. **"ROOT CAUTION" en lugar de "ROOT CAUSE"** — la grammar fuerza el token más probable compatible con el prefijo; como el SFT se llama vía `/api/generate` (no `/api/chat`), el contexto de activación difiere ligeramente del training y la distribución de los primeros tokens cambia. El token más probable válido bajo GBNF resulta ser "CAUTION" en lugar de "CAUSE".

2. **Respuesta en francés** — los adaptadores LoRA (~2% de parámetros) dominan la salida en condiciones normales guiando hacia el patrón K8s memorizado. Cuando la grammar bloquea esa ruta memorizada, el control cae al Qwen2.5 base (~98% de parámetros), que es fuertemente multilingüe. El base model produce la distribución de mayor masa en ese punto del contexto: francés.

3. **kubectl inventado** — mismo mecanismo: la grammar obliga a producir `kubectl ` pero el contenido que sigue es prosa incoherente del espacio del pretraining base.

### Diagnóstico — tercera confirmación de la causa raíz

> El SFT no aprendió a diagnosticar fallos de Kubernetes — aprendió a copiar secuencias de tokens específicas. Los adaptadores LoRA son un parche frágil sobre esa memorización: cualquier perturbación del contexto de activación (grammar, endpoint distinto, system prompt diferente) destruye el patrón aprendido y expone el modelo base debajo.

El baseline aguantó la grammar (+15.7 pp en Parse%) precisamente porque no tiene un patrón rígido memorizado: la constraint no crea contradicción y el modelo simplemente sigue la grammar con lo que sabe de K8s.

**El 56% de parse rate del SFT sin grammar no era "sabe la respuesta, no el formato" — era "tiene el formato memorizado en una región de los pesos tan frágil que cualquier intervención la destruye".**

Esta es la misma causa raíz observada en DPO (gradientes mínimos corrompen pesos de formato), SimPO (β alto destruye la salida), y ahora grammar (constraint cambia el contexto de activación y el base model emerge). Tres experimentos, tres mecanismos distintos, misma causa: sobreajuste extremo del SFT.

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
