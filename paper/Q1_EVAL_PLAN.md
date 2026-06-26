# Plan de evaluación para Q1 — checklist accionable

Objetivo: cerrar la brecha entre "system/experience report" y un paper Q1 aplicado
(IEEE TNSM / TSC, JSS, FGCS, Empirical SE, o congreso NOMS/ICSME/SoCC). La brecha
es de **evidencia empírica y encuadre**, no de mérito técnico. Cinco bloques (E1–E5),
ordenados por impacto. E1 y E3 son los imprescindibles.

Lo que ya tienes y reduce el esfuerzo:
- Harness de eval + 210 muestras ciegas (`eval/run_eval.py`, `eval/runner.py`, `eval/test_set.jsonl`).
- Backends de API (`anthropic`/`openai`/`ollama`) ya implementados en `src/diagnostics/escalation.py::_call_backend` → reutilizables para baselines SOTA.
- 14 escenarios de fallo YAML + generador (`dataset/`) → reutilizables para inyección de caos.
- Verificación por re-detección (Modo B) → ya mide resolución; solo falta instrumentar timestamps.
- Incident store → ya registra incidencias; añadir campos de tiempo.

---

## E1 — Baselines vs SOTA en diagnóstico (imprescindible)

**Por qué:** hoy solo te comparas con tus propias variantes. Un revisor Q1 exige
comparación con el estado del arte en el **mismo** test set.

**Qué correr** (sobre las 210 muestras ciegas, seed fija + bootstrap de E2):
- [ ] GPT-4o (API) — reusar `_call_backend("openai", ...)`.
- [ ] Claude Sonnet (API) — reusar `_call_backend("anthropic", ...)`.
- [ ] Un LLM local fuerte como referencia abierta (p. ej. `qwen2.5-coder:14b`, que ya está montado; o Llama-3.1-8B).
- [ ] Tus configuraciones: baseline 1.5B, ORPO+grammar, hybrid, **single-shot+digest (producción)**.

**Métricas:** Parse%, Keyword%, NS-ok%, Verb-ok%, latencia, **coste** (€/1k diagnósticos:
API vs CPU local — argumento on-prem) y la corrección humana de E2.

**Implementación:** añadir un backend "api" a `eval/runner.py` que llame a
`escalation._call_backend`; añadir los modelos al dict `MODELS` de `eval/run_eval.py`.
Tabla resultante = la tabla central del paper. **Tesis a demostrar:** el 1.5B en CPU
queda a X puntos de GPT-4o a 1/N del coste y sin egress de datos.

---

## E2 — Rigor estadístico + evaluación humana (soporta E1 y E3)

**Por qué:** una sola seed (99) y métricas propias (Keyword%/NS-ok%) no bastan; hay
que dar incertidumbre y un ground-truth humano.

- [ ] **Intervalos de confianza por bootstrap** sobre los 210 resultados (resampleo
      con reemplazo, 10k iteraciones → IC 95% de cada métrica). Reportar media ± IC.
- [ ] **≥3 seeds** en el split test (o en la generación del dataset) para varianza entre particiones.
- [ ] **Evaluación humana del diagnóstico:** muestrear N=100 diagnósticos, rúbrica
      (causa raíz correcta sí/parcial/no · comando accionable sí/no), **2–3 evaluadores**,
      reportar acuerdo inter-evaluador (Cohen/Fleiss κ). Esto sustituye el keyword-matching
      como verdad y responde a la crítica de "métricas propias".
- [ ] Correlacionar Keyword% con la corrección humana (validar la métrica automática).

**Entregable:** todas las tablas con IC; un apéndice de protocolo de anotación.

---

## E3 — Estudio end-to-end en producción: MTTR + detección + remediación (imprescindible, el de más impacto)

**Por qué:** es lo que más pesa en Q1 aplicado y hoy no existe. Mide el bucle entero,
no componentes sueltos.

**Montaje — inyección de caos** en el cluster real con ground-truth automático:
- [ ] Inyectar fallos conocidos y timestamped con un orquestador (chaos-mesh / litmus /
      scripts kubectl) usando tus 14 clases de `dataset/` (OOM, CrashLoop, ImagePull,
      probe, config, secret, NetworkPolicy, PVC, nodo, CPU…).
- [ ] **≥10 inyecciones por clase** → ~140 incidencias para estadística.
- [ ] Namespace de pruebas aislado (`aiops-demo`) para no tocar cargas reales.

**Qué medir (ground-truth = el fallo que inyectaste):**
- [ ] **Detección:** precision / recall / F1 vs fallos inyectados; **latencia de detección**
      (t_inyección → t_flag).
- [ ] **Diagnóstico:** corrección vs causa inyectada (automático).
- [ ] **Remediación:** tasa de éxito (¿resuelve, confirmado por re-detección/Modo B?),
      y por origen del plan (catálogo vs escalado agéntico).
- [ ] **MTTR**: t_inyección → t_resolución_verificada. Desglosar por:
      - manual (tú/un SRE resolviendo los mismos fallos sin el sistema) → **baseline de MTTR**,
      - con sistema HITL (aprobación por paso),
      - autónomo donde aplique (L1).
- [ ] **Falsos positivos en estado estacionario:** nº de alertas en una ventana sin caos.

**Implementación:** añadir campos `injected_at`/`resolved_at`/`detected_at` al incident
store; un script `eval/chaos_runner.py` que inyecte→espere→registre. **Tesis a demostrar:**
MTTR con-sistema << MTTR manual, con tasa de remediación verificada alta.

---

## E4 — Evaluación del planner agéntico

**Por qué:** hoy es esencialmente *un* ejemplo. Falta evaluación sistemática de calidad
y **seguridad** de los planes generados.

- [ ] Reunir M≥30 incidencias de *miss* del grafo (las clases investigate-only: image,
      pvc, node, pending_cpu, image_auth — inyectables vía E3).
- [ ] **Tasa de plan ejecutable**: % de miss en que el planner produce un `command` concreto
      (vs solo investigación + nota externa).
- [ ] **Resolución de placeholders**: % de comandos con nombres reales (no `<pod>`); comparar
      **single-shot escalation vs agéntico** (el diferencial que justifica el coste del 14B).
- [ ] **Seguridad (clave):** % de comandos generados que pasan el validador; % que serían
      destructivos si no se bloquearan; cero ejecuciones fuera del vocabulario seguro.
- [ ] **Éxito de remediación** de los planes escalados (re-detección) + calidad valorada por humano.
- [ ] Coste/latencia del escalado (carga on-demand del 14B en la A30).

---

## E5 — Reescritura con estructura de paper académico

**Por qué:** falta el andamiaje que un journal espera.

- [ ] **Research Questions** explícitas, p. ej.:
      - RQ1: ¿puede un SLM de 1.5B diagnosticar RCA de K8s de forma competitiva con SOTA en CPU?
      - RQ2: ¿remedia el grafo determinista + escalado agéntico de forma segura y efectiva?
      - RQ3: ¿reduce el bucle cerrado el MTTR frente a la operación manual?
- [ ] **Metodología** formal (dataset, splits, protocolo de anotación, montaje de caos).
- [ ] **Threats to validity** (internal/external/construct): un cluster, un operador,
      escenarios sintéticos, métricas; cómo los mitigas.
- [ ] **Related work** a fondo: detección de anomalías en logs, RCA por logs/trazas,
      LLM-for-ops (HolmesGPT, K8sGPT, Aurora), AIOps clásico — no solo 3 herramientas.
- [ ] **Reproducibilidad**: dataset + modelo ya en HuggingFace; añadir el harness de eval
      y los scripts de caos al release; *artifact evaluation* si el venue lo ofrece.

---

## Orden sugerido y esfuerzo aproximado

1. **E2** (bootstrap IC) — 1–2 días, desbloquea el rigor de E1/E3.
2. **E1** (baselines SOTA) — 2–3 días (harness ya casi listo).
3. **E3** (caos + MTTR) — 1–2 semanas (el de más valor; requiere montar inyección).
4. **E4** (planner) — se solapa con E3 (mismas inyecciones).
5. **E5** (reescritura) — 1 semana, en paralelo.

**Mínimo viable para Q1:** E1 + E2 + E3 (con MTTR vs manual) + E5. E4 lo refuerza.

## Venues objetivo
- **Journals Q1 aplicados:** IEEE TNSM, IEEE TSC, Journal of Systems and Software, Future Generation Computer Systems, Empirical Software Engineering.
- **Congresos fuertes (tracks aplicados/industry):** NOMS/IM, ICSME, SoCC, MLSys.
- **Mientras tanto:** arXiv preprint para fijar prioridad.
