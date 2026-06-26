# E1 — Baselines SOTA vs experto local (mismas 42 muestras)

Comparación apples-to-apples: mismo system prompt del experto, mismo parseo
`ROOT CAUSE/KUBECTL`, mismas métricas, **mismas 42 muestras** (3 por escenario ×
14 escenarios del test set ciego). IC 95% por bootstrap (10k iters, seed=99).

| Modelo | n | Parse% | Keyword% | NS-ok% (raw) | Verb-ok% | ROUGE-L | Lat. |
|---|---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o (API) | 42 | 97.6 [92.9, 100] | **100 [100, 100]** | 76.2 [61.9, 88.1] | 66.7 [52.4, 81.0] | 4.0 | 1.4 s |
| Claude Sonnet-4-6 (API) | 42 | 92.9 [83.3, 100] | 97.6 [92.9, 100] | 78.6 [66.7, 90.5] | 45.2 [31.0, 59.5] | 5.5 | 4.0 s |
| **k8s-rca-orpo 1.5B** (local, digest+grammar) | 42 | 95.2 [88.1, 100] | 76.2 [61.9, 88.1] | 52.4 [38.1, 66.7] | **71.4 [57.1, 83.3]** | **14.9** | **0.8 s** (GPU) |

*Latencias del local: ~0.8 s en GPU / ~32 s en CPU. APIs: latencia de red + GPU del proveedor.*

## Lectura honesta

- **En Keyword% (precisión diagnóstica) los modelos frontera ganan, y con
  significancia estadística.** GPT-4o 100 [100,100] y Claude 97.6 [92.9,100] vs el
  1.5B 76.2 [61.9,88.1] — los IC **no se solapan**. Es lo esperado: 100-1000× más
  parámetros. No lo ocultamos.
- **El 1.5B es competitivo en formato y se acerca en comando.** Parse% 95.2
  (garantizado por gramática, a la par de las APIs), gana en Verb-ok% (71.4, comandos
  alineados al catálogo) y en ROUGE-L (más cerca de las referencias). El NS-ok% **raw**
  (52.4) es bajo, pero es el comando *en bruto* del modelo: el **constructor determinista
  de comandos lo eleva a 85.7% en producción** (§17.7 del informe v1), neutralizando
  el gap de calidad de comando frente a las APIs.
- **El valor del 1.5B es de despliegue, no de tamaño:** coste marginal **≈0**
  (vs ~2-3 $/1k diagnósticos en GPT-4o/Claude), **sin egress** de las interioridades
  del cluster a un tercero, **viable en CPU** on-premise, y latencia ~0.8 s en GPU.

## Tesis para el paper

El 1.5B cede ~24 puntos de Keyword% frente a GPT-4o a cambio de **coste marginal cero,
cero egress de datos y viabilidad en CPU on-premise** — y la **capa de acción
determinista** (constructor de comandos + grafo) cierra el gap de calidad del comando
(NS-ok 52→86). Para operadores on-prem que no pueden enviar el estado del cluster a una
API externa, ese es el trade relevante: el modelo frontera diagnostica mejor en texto,
pero no se puede desplegar bajo esas restricciones.

## Reproducir

```bash
# APIs (claves por entorno; subconjunto 3/escenario)
OPENAI_API_KEY=…    python eval/run_api.py --provider openai    --model gpt-4o          --per-scenario 3
ANTHROPIC_API_KEY=… python eval/run_api.py --provider anthropic --model claude-sonnet-4-6 --per-scenario 3
# Local de producción (en el servidor con Ollama + A30)
python eval/run_api.py --provider ollama --model k8s-rca-orpo --per-scenario 3
# IC por bootstrap
python eval/bootstrap_ci.py <results.json>
```

*Pendiente para el paper final: ampliar a las 210 muestras y añadir la evaluación
humana (E2) como ground-truth del Keyword%.*
