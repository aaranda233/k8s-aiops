# SimPO Fine-tuning — Plan

**Fecha:** 2026-06-02
**Objetivo:** Superar el colapso del DPO v1 reutilizando el mismo dataset de 541 pares

---

## Por qué SimPO en lugar de DPO v2

El fallo del DPO v1 fue estructural: el SFT sobreajustado (loss=0.0898) y el
`ref_model` eran el mismo checkpoint, por lo que la log-ratio de referencia era ≈0
desde el paso 1. Cualquier ajuste de hiperparámetros (β mayor, 1 época) sólo mitigaría
el síntoma — el `ref_model` idéntico seguiría haciendo la señal mínima.

SimPO elimina el `ref_model` de la ecuación:

```
L = -log σ( β/|y_w| · log π_θ(y_w|x) − β/|y_l| · log π_θ(y_l|x) − γ )
```

La señal ahora viene de la **probabilidad absoluta** de cada respuesta normalizada por
su longitud, no de una ratio respecto a una referencia. Esto resuelve:

| Problema DPO v1 | Solución SimPO |
|----------------|----------------|
| Log-ratio ≈ 0 porque π_θ ≈ π_ref | No hay π_ref — señal siempre no nula |
| β=0.05 insuficiente como regularización KL | γ es un margen fijo escalar, no una penalización KL |
| Verbosidad (DPO premia respuestas largas) | Normalización por \|y\| quita la prima a la longitud |

---

## Implementación

### Stack
- **Trainer**: `trl.CPOTrainer` con `loss_type="simpo"` (TRL 0.11.4, disponible en servidor)
- **Base**: `k8s-rca-slm` (checkpoint SFT QLoRA 4-bit) — misma base que DPO v1
- **Dataset**: `dataset/output/dpo_dataset.jsonl` — 541 pares, sin regenerar
- **Script**: `finetune/train_simpo.py`

### Hiperparámetros clave

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `loss_type` | `"simpo"` | Activa normalización por longitud + elimina ref_model |
| `beta` | `2.0` | Valor del paper SimPO para tareas de instrucción |
| `simpo_gamma` | `1.0` | Margen objetivo — más alto que el default (0.5) para forzar separación real |
| `cpo_alpha` | `0.1` | Añade un término NLL pequeño para anclar el formato SFT durante training |
| `epochs` | `1` | 1 época evita el ruido acumulativo que destruyó el formato en DPO v1 |
| `lr` | `3e-5` | Más conservador que DPO v1 (5e-5) — el modelo SFT ya tiene el formato |
| `lora_r` | `16` | Mismo rango que DPO v1 |
| `batch_size` | `1` + grad_accum=`16` | Igual que DPO v1, ajustado para A30 24GB |

### El papel de `cpo_alpha`

`CPOTrainer` con SimPO tiene un término adicional opcional:

```
L_total = L_simpo + α · L_NLL(y_w)
```

Con `α=0.1`, el modelo recibe un pequeño gradiente supervisado sobre los `chosen`
en cada paso. Esto actúa como regularizador de formato — impide que la optimización
de preferencias destruya la estructura `ROOT CAUSE: / KUBECTL:` aprendida en el SFT,
sin dominar la señal de SimPO.

---

## Criterios de éxito

Evaluación con el mismo harness (210 muestras ciegas, seed=99):

| Métrica | SFT (baseline de comparación) | Objetivo SimPO |
|---------|:---:|:---:|
| Parse% | 56.2% | ≥ 56% (no regresión) |
| Keyword% | 60.0% | ≥ 75% (+15 pp) |
| ROUGE-L | 56.7% | ≥ 50% (toleramos pequeña regresión) |
| NS-ok% | 33.0% | ≥ 30% (no regresión) |

Si Keyword% sube y Parse% no cae, el experimento es un éxito.

---

## Pipeline completo

```bash
# 1. Entrenamiento (desde servidor)
python finetune/train_simpo.py

# 2. Registro en Ollama
cd finetune && ollama create k8s-rca-simpo -f Modelfile_simpo

# 3. Evaluación
python eval/run_eval.py --models sft,simpo,baseline
```

---

## Artefactos generados

| Archivo | Descripción |
|---------|-------------|
| `finetune/train_simpo.py` | Script de entrenamiento SimPO |
| `finetune/Modelfile_simpo` | Modelfile Ollama para k8s-rca-simpo |
| `finetune/output/k8s-rca-simpo/` | Adaptadores LoRA SimPO |
| `finetune/output/k8s-rca-simpo-Q8_0.gguf` | Modelo cuantizado para Ollama |
