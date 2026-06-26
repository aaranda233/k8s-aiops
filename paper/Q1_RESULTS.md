# Q1 — Resultados de evaluación (log vivo)

Seguimiento de los experimentos del plan [Q1_EVAL_PLAN.md](./Q1_EVAL_PLAN.md).
Estado: **E1 ✅ · E2 ✅ · E3 ✅(primer barrido + hallazgos) · E4 ✅ · E5 ✅**.
RQs, resultados, *threats to validity* y reproducibilidad integrados en
`RESEARCH.md` / `RESEARCH_es.md` (E5). Pendiente menor: barrido E3 estadístico
(N≥10/clase) en sistema templado y aislado, y evaluación humana del diagnóstico.

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

### MTTR — tiempo a remediación accionable

La parte de MTTR que el sistema **automatiza y se puede medir** es el **tiempo a un
plan de remediación accionable** = detección + diagnóstico + plan listo. Medido en el
barrido: **~15-36 s** desde la inyección del fallo hasta la incidencia con su plan
multi-paso validado. El resto del MTTR (aprobación humana + ejecución + verificación)
lo automatizan el ejecutor por-paso y la re-detección (Modo B).

Comparación vs manual: el triaje manual en K8s (notar la alerta → investigar →
diagnosticar) suele tomar minutos a decenas de minutos; un MTTR-vs-manual **empírico**
(con un baseline humano y fallos auto-resolubles diseñados) requiere un estudio
controlado → E5/futuro. *No reclamamos un número manual medido aquí.*

### Hallazgo operativo — contención pipeline ↔ escalado 14B

Al encadenar E4 (investigaciones repetidas del coder-14b) con un barrido E3, el
**bucle del pipeline se quedó colgado** (sin excepción; última ventana 12:37, cero
detecciones durante ~20 min). El pipeline es de **un solo hilo** y comparte GPU/Ollama
con el modelo grande on-demand; bajo escalado agéntico intenso la llamada de
diagnóstico puede quedar en cola y **starve** la detección. Se recuperó con un
reinicio. **Es una limitación real** (concurrencia/aislamiento de recursos) que va a
*threats to validity* y motiva: cola/aislamiento entre detección y escalado, y no
solapar cargas de eval. *(Por esto los experimentos de cluster deben correr de uno en
uno con el sistema "templado", no en ráfaga.)*

### Pendiente E3
- [ ] Barrido mayor (N≥10/clase) **de uno en uno, sin solapar con E4**, para recall/latencia con IC.
- [ ] Arreglar/ajustar la detección de fallos solo-eventos (image-pull/PVC/node).
- [ ] MTTR-vs-manual empírico (estudio controlado, E5).
- [ ] Precisión de detección (falsos positivos sin caos).

---

## E4 — Eval del planner agéntico ✅

`eval/eval_planner.py` — para cada clase, inyecta el fallo en un namespace vivo y
compara, con el **mismo** modelo grande (`qwen2.5-coder:14b`), single-shot (llamada
ciega) vs agéntico (investiga en read-only y luego propone). 3 clases × 2 = 6.

| Modo | Ejecutable | Sin placeholder | Seguro | Latencia |
|---|:---:|:---:|:---:|:---:|
| single-shot | 5/6 | 5/6 | 5/6 | 5-11 s |
| **agéntico** | 5/6 | **6/6** | **6/6** | 7-28 s |

**Hallazgo:** el agéntico **elimina los placeholders y es siempre seguro** (6/6 vs
5/6). El caso revelador es `image #0`: el single-shot produjo un plan **inválido con
placeholder** (`exec=False`, 0 pasos), mientras el agéntico, investigando el cluster
en vivo (28 s), emitió un plan ejecutable de 4 pasos con nombres reales. El coste es
latencia (investigación read-only): 7-28 s vs 5-11 s del ciego. Cuando el agéntico no
halla acción segura devuelve investigate-only (p. ej. `oom #1`) en vez de inventar —
por diseño. Detalle: `eval/results/e4_planner.json`.

**Lectura:** el agéntico cambia latencia por **fiabilidad y seguridad** — justo donde
el ciego falla (resolución de nombres reales en la cola larga). Justifica el coste del
modelo de 14B on-demand para el escalado.

---

## E5 — Reescritura académica ✅

Integrado en `RESEARCH.md` y `RESEARCH_es.md`:
- **Research Questions** explícitas (RQ1 diagnóstico, RQ2 remediación, RQ3 bucle en
  producción/MTTR, RQ4 coste del escalado) en la Introducción.
- **Resultados reorganizados por RQ**, con las tablas E1 (baselines + IC), E2 (IC),
  E4 (planner) y E3 (detección/MTTR).
- **Threats to validity** (constructo, externa, interna=contención, recall) y
  **Reproducibilidad** (scripts seedados + modelos/dataset en HF).
- Limitaciones y trabajo futuro actualizados (E1/E3 ya hechos).

> Nota operativa (E3): el caos justo tras un reinicio cae en el **warm-up de novedad**
> del detector y no dispara. Los barridos válidos exigen el sistema **templado** y sin
> escalado concurrente (ver contención arriba).
