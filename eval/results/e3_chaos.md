# E3 — Inyección de caos en producción (primer barrido validado)

Inyección de fallos conocidos en namespaces aislados del cluster real (uno por
inyección, para evitar la deduplicación por namespace), midiendo **latencia de
detección** y **corrección del diagnóstico** (keyword vs el fallo inyectado).
`eval/chaos_runner.py`, 3 clases × 2 repeticiones.

| Fallo | Clase | Detectado | Latencia | Diagnóstico (keyword) |
|---|---|:---:|:---:|:---:|
| crashloop #0 | crash_config | ✓ | 26 s | ✓ |
| crashloop #1 | crash_config | ✓ | 36 s | ✓ |
| oom #0 | crash_oom | ✓ | 15 s | ✓ |
| oom #1 | crash_oom | ✗ (timeout 200 s) | — | — |
| image #0 | image_not_found | ✗ (timeout 200 s) | — | — |
| image #1 | image_not_found | ✗ (timeout 200 s) | — | — |

**Detección:** 3/6 (50%). **Latencia de detección (detectados):** media 26 s,
máx 36 s — muy por debajo del presupuesto de ventana de 60 s. **Diagnóstico:**
3/3 correcto (keyword) sobre los detectados.

## Lectura honesta

- **Cuando se detecta, es rápido y el diagnóstico acierta** (15-36 s, 3/3 keyword).
  La latencia de detección end-to-end queda dentro del presupuesto.
- **La recall varía por clase de fallo.** crashloop (2/2) y oom (1/2) se detectan;
  **image-pull (0/2) no.** Hipótesis: los fallos *solo-eventos* (ImagePullBackOff
  no arranca el contenedor → **cero logs de pod**, solo eventos K8s escasos) generan
  poca señal en un namespace nuevo y diminuto, y no cruzan el umbral de anomalía —
  mientras que crashloop/oom sí producen logs. Es un hallazgo, no un bug: motiva
  ajustar la sensibilidad de detección para fallos pobres en señal.
- **Caveat de los workloads sintéticos:** busybox produce evidencia pobre, así que
  la prosa del diagnóstico es genérica (en prod, con workloads reales como
  PostgreSQL, es específica). Para el barrido final conviene usar fallos más realistas.

## Pendiente para el E3 completo (paper)

- **Recall por clase con N≥10/clase** + investigar/ajustar la detección de
  image-pull/PVC/node (clases solo-eventos).
- **MTTR vs manual:** medir t_inyección → t_resolución_verificada con el bucle de
  remediación (HITL y autónomo L1) vs un baseline manual. Esto **ejecuta acciones**
  en el cluster → paso deliberado aparte, pendiente de OK.
- **Precisión de detección** (falsos positivos en ventana sin caos).

## Reproducir

```bash
# en el servidor (kubectl + app viva + A30)
python eval/chaos_runner.py --fault all --repeat 2 --timeout 200 --out e3_chaos.json
```
