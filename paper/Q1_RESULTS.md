# Q1 — Resultados de evaluación (log vivo)

Seguimiento de los experimentos del plan [Q1_EVAL_PLAN.md](./Q1_EVAL_PLAN.md).
Se actualiza conforme avanzamos. Estado: **E1 ✅ · E2 ✅ · E3 🚧 · E4 🚧 · E5 ⏳**.

Reproducibilidad: todos los runs con seed=99; IC95 por bootstrap (10k iters).
Artefactos en `eval/results/`.

---

## E2 — Rigor estadístico (IC por bootstrap) ✅

`eval/bootstrap_ci.py` calcula media ± IC95 (percentil) por modelo/métrica desde
el `per_sample` de un run. Sobre el run canónico (210 muestras, ORPO vs Híbrido):

| Modelo | Keyword% | NS-ok% | Parse% |
|---|:---:|:---:|:---:|
| ORPO (single) | 78.1 [72.4, 83.8] | **89.5 [85.2, 93.3]** | 100 |
| Híbrido | **92.9 [89.0, 96.2]** | 73.3 [67.1, 79.0] | 98.6 |

**Hallazgo:** los IC de Keyword% (gana híbrido) y de NS-ok% (gana single-shot) **no se
solapan** → ambas diferencias son estadísticamente significativas.

---

## E1 — Baselines SOTA vs experto local ✅

`eval/run_api.py` — mismas 42 muestras (3/escenario), mismo prompt y métricas, IC95.

| Modelo | Keyword% | NS-ok% (raw) | Parse% | Lat. |
|---|:---:|:---:|:---:|:---:|
| GPT-4o (API) | **100 [100,100]** | 76.2 [61.9,88.1] | 97.6 | 1.4 s |
| Claude Sonnet-4-6 (API) | 97.6 [92.9,100] | 78.6 [66.7,90.5] | 92.9 | 4.0 s |
| k8s-rca-orpo 1.5B (local) | 76.2 [61.9,88.1] | 52.4 [38.1,66.7] | 95.2 | 0.8 s |

**Hallazgo:** los modelos frontera baten al 1.5B en Keyword% **con significancia**
(IC disjuntos). El valor del 1.5B es de despliegue: coste marginal ~0 (vs ~2-3 $/1k),
sin egress, CPU-viable; y el constructor determinista cierra el gap de comando
(NS-ok raw 52% → 86% en producción). Detalle: `eval/results/e1_baselines.md`.

---

## E3 — Caos en producción 🚧

Inyección de fallos conocidos en namespaces aislados (uno por inyección, para evitar
la dedup por namespace). `eval/chaos_runner.py`. Cluster real (A30).

### Detección + diagnóstico (barrido 3 clases × 2)

| Fallo | Detectado | Latencia | Diagnóstico (keyword) |
|---|:---:|:---:|:---:|
| crashloop ×2 | 2/2 | 26 s, 36 s | 2/2 ✓ |
| oom ×2 | 1/2 | 15 s | 1/1 ✓ |
| image ×2 | 0/2 | timeout | — |

- **Detección rápida y diagnóstico correcto cuando dispara**: latencia 15-36 s
  (< 60 s de presupuesto), 3/3 keyword sobre los detectados.
- **Hallazgo — recall por clase:** los fallos *solo-eventos* (image-pull: no arranca
  el contenedor → cero logs de pod, eventos escasos) se detectan peor que los que
  producen logs (crashloop/oom). Motiva ajustar la sensibilidad para clases pobres
  en señal. Detalle: `eval/results/e3_chaos.md`.

### Pendiente E3
- [ ] Barrido mayor (N≥10/clase) para recall/latencia con IC.
- [ ] MTTR: tiempo a plan accionable (auto) vs triaje manual.
- [ ] Precisión de detección (falsos positivos sin caos).

---

## E4 — Eval del planner agéntico 🚧

*(en progreso — se rellena al terminar el run)*

---

## E5 — Reescritura académica ⏳

Pendiente: RQs, metodología, threats to validity, related work a fondo.
