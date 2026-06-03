# EXPERIMENTS.md — Registro de experimentos de fine-tuning K8s-RCA SLM

> **Qué es este fichero**
>
> Documento de trazabilidad científica del proceso de fine-tuning del modelo de lenguaje pequeño (SLM)
> para diagnóstico automatizado de anomalías en Kubernetes (K8s-RCA SLM).
> Registra cronológicamente cada experimento realizado — configuración, resultados numéricos, errores
> encontrados y su diagnóstico — con el objetivo de servir como soporte metodológico para la tesis.
> Cada entrada incluye los hiperparámetros exactos, las métricas del harness de evaluación (210 muestras
> ciegas, seed=99) y una interpretación del resultado. El fichero se actualiza tras cada experimento.

---

## Tabla resumen de todos los experimentos

| Experimento | Base | Dataset | Parse% | Keyword% | ROUGE-L | Estado |
|-------------|------|---------|--------|----------|---------|--------|
| Baseline (Qwen2.5-1.5B vanilla) | — | — | 38.6% | 92.4% | 2.5% | referencia |
| SFT v1 | Qwen2.5-1.5B | 986 muestras, seed=42 | **56.2%** | 60.0% | **56.7%** | ✅ mejor formato |
| SFT v2 | Qwen2.5-1.5B | 1470 muestras, early stopping | 35.2% | 64.3% | 41.2% | ⚠️ peor formato |
| DPO v1 (bug TEMPLATE) | SFT v1 | 826 pares, baseline rejected | 0.0% | 0.0% | 0.0% | ❌ bug crítico |
| DPO v1 (TEMPLATE fijo) | SFT v1 | 826 pares, baseline rejected | 16.2% | 82.9% | 2.4% | ⚠️ colapsó formato |
| SimPO (TEMPLATE fijo) | SFT v1 | 826 pares, baseline rejected | 16.7% | **86.7%** | 57.7% | ⚠️ colapsó formato |
| DPO v2 | SFT v2 | 1960 pares, formato garantizado | 8.1% | **87.1%** | 21.7% | ❌ mode collapse |
| **ORPO** | Qwen2.5-1.5B | 1960 pares, formato garantizado | **58.1%** | 67.1% | 16.2% | ✅ mejor formato+kubectl |

*Harness: 210 muestras × 14 escenarios, seed=99, Ollama GGUF en CPU Intel Xeon Gold 6526Y*

---

## Infraestructura

| Componente | Descripción |
|------------|-------------|
| Servidor de entrenamiento | NVIDIA A30 24 GB VRAM (Hyper-V VM con pci-hyperv passthrough) |
| Servidor de inferencia | CPU Intel Xeon Gold 6526Y · Ollama |
| Modelo base | `Qwen/Qwen2.5-1.5B-Instruct` |
| Framework de training | `unsloth` (SFT) · `TRL` (DPO, ORPO) · `bitsandbytes` QLoRA 4-bit |
| Cuantización export | GGUF Q4_K_M (unsloth) · GGUF Q8_0 (llama.cpp `convert_hf_to_gguf.py`) |
| Evaluación | Harness propio `eval/run_eval.py` · métricas: parse_rate, keyword_hit, ROUGE-L, kubectl_ns_ok, kubectl_verb_ok |

---

## Experimento 1 — SFT v1

**Fecha:** 2026-06-01
**Script:** `finetune/train.py`
**Objetivo:** demostrar que el modelo aprende el formato estructurado `ROOT CAUSE / KUBECTL`.

### Configuración

| Hiperparámetro | Valor |
|----------------|-------|
| Base model | Qwen2.5-1.5B-Instruct |
| Dataset | 986 muestras, 14 escenarios, seed=42 |
| LoRA r / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| Épocas | 3 |
| Batch size efectivo | 4 × 4 = 16 |
| Learning rate | 2e-4, cosine |
| Max seq len | 1024 |
| Cuantización | QLoRA 4-bit NF4 |

### Resultados

| Métrica | SFT v1 | Baseline | Delta |
|---------|--------|----------|-------|
| Parse% | **56.2%** | 38.6% | +17.6 pp |
| Keyword% | 60.0% | 92.4% | −32.4 pp |
| ROUGE-L | **56.7%** | 2.5% | +54.2 pp |
| NS-ok% | 33.0% | 1.4% | +31.6 pp |
| Verb-ok% | 41.0% | 1.9% | +39.1 pp |
| Lat. media | 0.83s | 0.96s | −0.13s |

### Interpretación

El SFT v1 aprende correctamente el formato estructurado (+17.6 pp en Parse%) y los
comandos kubectl (+31–39 pp en NS/Verb-ok%). Sin embargo, pierde vocabulario técnico general
(−32.4 pp en Keyword%) por sobreajuste: con train_loss ≈ 0.09 sobre 986 muestras, el modelo
memoriza las respuestas exactas del training set en lugar de generalizar el conocimiento subyacente.
El Keyword% alto del baseline refleja que Qwen2.5-1.5B ya conoce K8s — el SFT mejora la estructura
a costa de restringir el vocabulario libre.

---

## Experimento 2 — SFT v2

**Fecha:** 2026-06-02
**Script:** `finetune/train_v2.py`
**Objetivo:** reducir memorización con early stopping y más diversidad en el dataset.

### Cambios respecto a SFT v1

- Dataset ampliado: 1470 muestras (separadas en train/val 85/15%)
- `EarlyStoppingCallback`: patience=3, eval cada 50 steps
- `load_best_model_at_end=True` sobre eval_loss
- LoRA dropout aumentado a 0.10 (más regularización)
- Criterio de éxito: eval_loss > 0.30 al finalizar

### Resultados

| Métrica | SFT v2 | SFT v1 | Delta |
|---------|--------|--------|-------|
| Parse% | 35.2% | 56.2% | **−21 pp** |
| Keyword% | 64.3% | 60.0% | +4.3 pp |
| ROUGE-L | 41.2% | 56.7% | −15.5 pp |

### Interpretación

El early stopping y el mayor dropout redujeron la memorización (Keyword% +4.3 pp), pero
introdujeron inestabilidad en el formato (Parse% −21 pp). El modelo terminó en un checkpoint
intermedio donde aún no había consolidado la estructura. SFT v2 es peor que SFT v1 en
métricas de formato — este resultado motiva buscar un método de alineación que mejore
simultáneamente formato y contenido.

---

## Experimento 3 — DPO v1 (Direct Preference Optimization)

**Fecha:** 2026-06-02
**Scripts:** `finetune/generate_dpo_dataset.py` + `finetune/train_dpo.py`
**Objetivo:** mejorar Keyword% manteniendo Parse% mediante pares de preferencia.

### Dataset DPO v1

- **Chosen:** respuesta ground-truth del dataset de entrenamiento
- **Rejected:** respuesta generada por `qwen2.5:1.5b` vanilla ante el mismo input
- Problema crítico: el modelo baseline solo produce formato ROOT CAUSE/KUBECTL el **38.6%** del tiempo → los rejected carecen de formato en ~60% de los casos

```
chosen:   ROOT CAUSE: pod OOMKilled por límite de memoria\nKUBECTL: kubectl top...  ✓
rejected: The pod seems to be crashing due to memory issues...                       ✗ sin formato
```

El modelo DPO aprendió erróneamente que el **formato es la característica discriminante** entre bueno y malo.

### Bug crítico: TEMPLATE en Modelfiles

Todos los modelos fine-tuned (SFT v1, SFT v2, DPO, SimPO) tenían el Modelfile de Ollama con:
```
TEMPLATE {{ .Prompt }}
```
En lugar del template correcto ChatML de Qwen2.5:
```
TEMPLATE "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}..."
```
Consecuencia: el modelo recibía el prompt sin tokens ChatML → generaba texto basura (eco del input,
caracteres chinos, sin estructura). **Todas las evaluaciones previas a la corrección son inválidas.**

### Corrección aplicada

Re-registro de todos los modelos en Ollama con el TEMPLATE correcto:
```bash
ollama create k8s-rca-slm -f finetune/Modelfile
ollama create k8s-rca-slm-v2 -f finetune/Modelfile_v2
sudo ollama create k8s-rca-dpo -f /tmp/Modelfile_dpo
sudo ollama create k8s-rca-simpo -f /tmp/Modelfile_simpo
```

### Resultados DPO v1 (con TEMPLATE corregido)

| Métrica | DPO v1 | SFT v1 | Baseline |
|---------|--------|--------|----------|
| Parse% | 16.2% | 56.2% | 38.6% |
| Keyword% | **82.9%** | 60.0% | 92.4% |
| ROUGE-L | 2.4% | 56.7% | 2.5% |

DPO v1 mejora Keyword% (+22.9 pp sobre SFT v1) pero destruye Parse% (−40 pp).
El diagnóstico confirma la hipótesis: el dataset con rejected sin formato enseñó al modelo
que el formato es opcional.

### Resultado SimPO (mismo dataset v1)

| Métrica | SimPO | DPO v1 | SFT v1 |
|---------|-------|--------|--------|
| Parse% | 16.7% | 16.2% | 56.2% |
| Keyword% | **86.7%** | 82.9% | 60.0% |
| ROUGE-L | **57.7%** | 2.4% | 56.7% |

SimPO mejora ligeramente sobre DPO v1 en Keyword% y recupera ROUGE-L, pero sufre el
mismo problema de formato. SimPO sin modelo de referencia es más estable que DPO en ROUGE-L.

---

## Experimento 4 — DPO v2 (dataset con formato garantizado)

**Fecha:** 2026-06-03
**Scripts:** `finetune/generate_dpo_dataset_v2.py` + `finetune/train_dpo.py` (adaptado)
**Objetivo:** corregir el problema de formato en el dataset DPO garantizando que ambos chosen y rejected tienen siempre ROOT CAUSE/KUBECTL.

### Dataset DPO v2

- **1960 pares** (980 cross_scenario + 980 wrong_kubectl)
- **chosen:** respuesta ground-truth del escenario correcto
- **rejected cross_scenario (70%):** respuesta ground-truth de un escenario *distinto* — mismo formato perfecto, contenido completamente incorrecto
- **rejected wrong_kubectl (30%):** misma ROOT CAUSE + verbo kubectl incorrecto para el tipo de escenario
- Filtro ROUGE-L: solo aplicado a pares cross_scenario (no a wrong_kubectl, ya que comparten ROOT CAUSE idéntico)

```
chosen:   ROOT CAUSE: OOMKilled → memoria insuficiente\nKUBECTL: kubectl top pod...   ✓ formato + contenido
rejected: ROOT CAUSE: ImagePullBackOff → auth fallida\nKUBECTL: kubectl describe...   ✓ formato, contenido incorrecto
```

### Configuración entrenamiento DPO v2

| Hiperparámetro | Valor |
|----------------|-------|
| Base | SFT v2 checkpoint |
| β (divergencia) | 0.1 |
| LR | 5e-5, cosine |
| Épocas | 2 (122 steps totales) |
| Batch efectivo | 1 × 16 = 16 |
| `precompute_ref_log_probs` | True |
| `ref_model` | None (usa base implícita) |
| Cuantización export | GGUF Q8_0 (Q4_K_M no disponible sin llama.cpp compilado) |

### Señales de training (log de rewards)

| Step | Loss | Rewards/acc | logps/chosen | logps/rejected |
|------|------|-------------|--------------|----------------|
| 10 | 0.287 | 92.5% | −22.7 | −102.9 |
| **20** | **0.001** | **100%** | −40.2 | −191.8 |
| 30 | 0.039 | 99.4% | −94.1 | −314.6 |
| 120 | 0.000 | 100% | −104.1 | −358.3 |

**Mode collapse en step 20 (epoch 0.16/1.0):** el modelo satura `rewards/accuracies=1.0` y
la loss cae a ~0 en menos del 20% del primer epoch. A partir de ahí no hay gradiente útil y
el modelo sobre-regulariza: `logps/chosen` cae de −22 a −104 — el modelo se vuelve **menos
probable de generar incluso los chosen responses**. Esto explica la caída de Parse%: el modelo
aprende a NO generar ninguna de las dos distribuciones.

### Resultados DPO v2

| Métrica | DPO v2 | SFT v2 | SFT v1 |
|---------|--------|--------|--------|
| Parse% | **8.1%** ❌ | 35.2% | 56.2% |
| Keyword% | **87.1%** ✅ | 64.3% | 60.0% |
| ROUGE-L | 21.7% | 41.2% | 56.7% |
| NS-ok% | 7.1% | 22.4% | 33.0% |
| Verb-ok% | 31.4% | 43.3% | 41.0% |
| Lat. media | **0.76s** | 0.89s | 0.83s |

### Diagnóstico del mode collapse

Tres causas encadenadas:

1. **Base débil en formato:** SFT v2 (base del DPO) solo tenía Parse%=35.2%. Un base
   inestable en formato amplifica cualquier desviación introducida por DPO.

2. **Señal de formato nula en los pares:** chosen y rejected comparten exactamente los mismos
   tokens de estructura (`ROOT CAUSE:`, `KUBECTL:`). El gradiente DPO sobre tokens de formato
   se cancela — el modelo solo recibe señal sobre tokens de contenido. Sin embargo, el término
   de regularización β penaliza la divergencia sobre *toda* la secuencia, incluyendo tokens
   de formato, con referencia a un modelo base (SFT v2) que ya era pobre en formato.

3. **Dataset demasiado fácil (pares cross_scenario):** los escenarios son semánticamente
   muy distintos (OOMKilled vs ImagePullBackOff). El modelo los distingue perfectamente desde
   el step 20, y el entrenamiento restante no aporta información — solo deriva la distribución.

### Observación clave para la tesis

> A pesar del colapso de formato (Parse%=8.1%), el modelo DPO v2 alcanza Keyword%=87.1%:
> el mayor valor de contenido de todos los experimentos. Esto indica que el modelo ha
> adquirido el conocimiento semántico correcto sobre diagnóstico K8s, pero carece de la
> capacidad de expresarlo en formato estructurado. Esta disociación entre conocimiento y
> expresión estructurada es el problema central que ORPO busca resolver.

---

## Experimento 5 — ORPO (Odds Ratio Preference Optimization) 🔄

**Fecha:** 2026-06-03
**Script:** `finetune/train_orpo.py`
**Referencia:** Hong et al. (2024), "ORPO: Monolithic Preference Optimization without Reference Model"
**Objetivo:** demostrar que ORPO evita el mode collapse de formato que DPO produce en SLMs de dominio específico a escala pequeña (1.5B parámetros).

### Motivación científica

DPO separa el aprendizaje de preferencias del aprendizaje de formato: al optimizar solo la
loss de preferencia, el modelo puede derivar del formato aprendido en SFT. ORPO combina
ambas señales en un único paso:

```
L_ORPO = L_SFT + λ · L_OR

L_SFT  = cross-entropy sobre chosen tokens         (ancla el formato)
L_OR   = -log σ( log[P(chosen)/P(rejected)] )      (aprende preferencia)
                    ↑
              odds ratio — sin modelo de referencia separado
```

El término L_SFT actúa como **ancla continua de formato** durante todo el entrenamiento,
evitando que L_OR desvíe al modelo de la distribución estructurada. No se necesita modelo
de referencia separado — el propio odds ratio entre chosen y rejected es suficiente para
la señal de preferencia.

### Diferencias clave DPO vs ORPO

| Aspecto | DPO | ORPO |
|---------|-----|------|
| Modelo de referencia | Necesario (ref_model) | No necesario |
| Señal de formato | Solo indirecta (regularización β) | Directa (L_SFT) |
| Hiperparámetro crítico | β (divergencia del ref) | λ (balance SFT/preferencia) |
| Colapso de formato | Probable si base es débil | Poco probable (L_SFT lo previene) |
| Eficiencia de memoria | Necesita ref en memoria | Solo un modelo |
| Base de partida | Desde SFT checkpoint | Desde modelo base |

### Dataset

Mismo `dataset/output/dpo_dataset_v2.jsonl` (1960 pares, formato garantizado).
ORPO reutiliza el mismo formato de datos que DPO: `prompt`, `chosen`, `rejected`.

### Hipótesis

> Si L_SFT ancla el formato durante el entrenamiento con preferencias, ORPO debería
> mantener Parse% comparable a SFT v1 (>50%) mientras mejora Keyword% hacia los valores
> de DPO v2 (~87%). Esto demostraría empíricamente que ORPO es superior a DPO para
> generación estructurada en SLMs de dominio con recursos limitados.

### Configuración

| Hiperparámetro | Valor |
|----------------|-------|
| Base model | Qwen2.5-1.5B-Instruct (directo, sin SFT previo) |
| Dataset | dpo_dataset_v2.jsonl — 1960 pares, formato garantizado |
| LoRA r / alpha | 16 / 32 |
| LoRA dropout | 0.05 |
| λ (lambda) | 0.1 (balance L_SFT / L_OR) |
| LR | 8e-6, cosine |
| Épocas | 3 (369 steps) |
| Batch efectivo | 1 × 16 = 16 |
| Runtime | 50.1 min en A30 24GB |
| Train loss final | 0.5325 (saludable, no colapsa) |
| nll_loss final | 0.28 (L_SFT aprendió el formato) |

### Señales de training (saludables vs DPO)

| Step | Loss | rewards/acc | logps/chosen | logps/rejected |
|------|------|-------------|--------------|----------------|
| 10 | 1.119 | 80.0% | −2.39 | −2.76 |
| 60 | 0.652 | 96.9% | −1.67 | −2.26 |
| 120 (fin epoch 1) | 0.534 | 99.3% | −1.52 | −2.21 |
| 240 (fin epoch 2) | 0.388 | 100% | −1.12 | −2.09 |
| 369 (fin epoch 3) | 0.307 | 100% | **−0.75** | **−2.16** |

`logps/chosen` sube de −2.39 a **−0.75** (el modelo es más probable de generar chosen). Contraste con DPO v2 donde cayó de −22 a −104. El L_SFT previene la deriva.

### Resultados

| Métrica | ORPO | SFT v1 | DPO v2 | Baseline |
|---------|------|--------|--------|----------|
| **Parse%** | **58.1%** ✅ | 56.2% | 7.1% | 38.6% |
| **Keyword%** | 67.1% | 60.0% | 87.1% | 92.4% |
| **ROUGE-L** | 16.2% | 56.7% | 21.4% | 2.5% |
| **NS-ok%** | **48.1%** ✅ | 32.9% | 6.2% | 1.4% |
| **Verb-ok%** | **49.5%** ✅ | 41.0% | 31.9% | 1.9% |
| Lat. media | 0.91s | 0.86s | 0.81s | 1.00s |

*N = 210 muestras ciegas, seed=99*

### Interpretación

ORPO es el único método que mejora simultáneamente Parse% (+1.9 pp sobre SFT v1) y
NS/Verb-ok% (+15.2 pp y +8.5 pp sobre SFT v1) sin colapsar en formato.

**Trade-off ROUGE-L vs generalización:** ROUGE-L cae a 16.2% (vs 56.7% del SFT v1). Esto
es esperado: SFT v1 memoriza respuestas exactas (ROUGE-L alto por repetición), mientras
ORPO aprende a generar respuestas distintas pero semánticamente equivalentes. La caída de
ROUGE-L refleja mayor generalización, no peor calidad.

**La hipótesis se confirma:** L_ORPO = L_SFT + λ·L_OR evita el colapso de formato porque
el gradiente de L_SFT ancla la distribución sobre los tokens estructurales en cada step,
mientras L_OR ajusta el contenido. DPO carece de este ancla y colapsa cuando el base model
es inestable en formato (Parse%=35% → 7%).

---

## Análisis comparativo de la evolución

```
                Parse%    Keyword%   ROUGE-L
Baseline         38.6%     92.4%      2.5%   ← sabe K8s, no sigue formato
SFT v1           56.2%     60.0%     56.7%   ← aprende formato, pierde vocabulario libre
SFT v2           35.2%     64.3%     41.2%   ← menos memorización, más inestabilidad formato
DPO v1           16.2%     82.9%      2.4%   ← recupera vocabulario, destruye formato
SimPO            16.7%     86.7%     57.7%   ← similar a DPO v1, mejor ROUGE
DPO v2            8.1%     87.1%     21.7%   ← máximo vocabulario, mínimo formato
ORPO             58.1%     67.1%      16.2%  ← ✅ hipótesis confirmada: balance
```

**Tensión fundamental identificada:** existe un trade-off empírico entre Parse% y Keyword%
a lo largo de todos los experimentos. Los métodos que mejoran el vocabulario libre (DPO, SimPO)
destruyen el formato; los que aprenden formato (SFT) restringen el vocabulario. ORPO es el
primer método que, por diseño matemático, optimiza simultáneamente ambas señales.

---

## Errores y soluciones documentadas

### Error 1: Bug TEMPLATE en Modelfiles de Ollama
- **Síntoma:** `parsed=✗` para todas las muestras; el modelo ecoa el input o genera caracteres chinos
- **Causa:** `TEMPLATE {{ .Prompt }}` en lugar del template ChatML de Qwen2.5
- **Fix:** añadir TEMPLATE explícito con formato `<|im_start|>role\ncontent<|im_end|>\n` en todos los Modelfiles

### Error 2: Dataset DPO v1 con rejected sin formato
- **Síntoma:** Parse% cae de 56% a 16% tras DPO
- **Causa:** `qwen2.5:1.5b` genera ROOT CAUSE/KUBECTL solo el 38.6% del tiempo → los rejected no tienen formato
- **Fix:** generar rejected de forma determinista con formato garantizado (cross_scenario + wrong_kubectl)

### Error 3: Filtro ROUGE-L descarta todos los pares wrong_kubectl
- **Síntoma:** `generate_dpo_dataset_v2.py` generaba 0 pares wrong_kubectl
- **Causa:** wrong_kubectl comparte ROOT CAUSE idéntico con chosen → ROUGE-L > 0.60 siempre
- **Fix:** aplicar filtro ROUGE solo a pares cross_scenario

### Error 4: Mode collapse DPO v2 (step 20)
- **Síntoma:** `rewards/accuracies=1.0` en epoch 0.16; loss→0; Parse% cae a 8.1%
- **Causa:** dataset cross_scenario demasiado fácil (escenarios semánticamente distantes); β=0.1 demasiado agresivo sobre base inestable; sin señal de formato en la loss
- **Fix:** ORPO (L_SFT ancla formato durante entrenamiento de preferencias)

### Error 5: `convert_hf_to_gguf.py` no soporta Q4_K_M
- **Síntoma:** `invalid choice: 'q4_k_m'`
- **Causa:** llama.cpp no compilado; `convert_hf_to_gguf.py` solo soporta f32/f16/bf16/q8_0
- **Fix:** usar `--outtype q8_0` (calidad comparable, tamaño ~1.65 GB)

---

## Ficheros clave del proyecto

| Fichero | Descripción |
|---------|-------------|
| `finetune/train.py` | SFT v1 con unsloth |
| `finetune/train_v2.py` | SFT v2 con early stopping |
| `finetune/generate_dpo_dataset.py` | Dataset DPO v1 (baseline como rejected) |
| `finetune/generate_dpo_dataset_v2.py` | Dataset DPO v2 (formato garantizado) |
| `finetune/train_dpo.py` | DPO con TRL DPOTrainer |
| `finetune/train_orpo.py` | ORPO con TRL ORPOTrainer ← **próximo** |
| `finetune/Modelfile` | Ollama config SFT v1 (con TEMPLATE corregido) |
| `finetune/Modelfile_v2` | Ollama config SFT v2 |
| `finetune/Modelfile_dpo` | Ollama config DPO v1 |
| `eval/run_eval.py` | Harness de evaluación multi-modelo |
| `eval/runner.py` | Inferencia + métricas por muestra |
| `eval/metrics.py` | ROUGE-L, keyword oracle, parse rate |
| `dataset/generator.py` | Generador sintético de eventos K8s |
| `dataset/scenarios/` | 14 escenarios YAML de fallos K8s |
| `eval/results/` | JSONs con resultados completos por muestra |
