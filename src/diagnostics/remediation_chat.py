"""Chat de remediación conversacional.

El operador conversa con el modelo grande (p. ej. el Qwen del GB10, configurado
como backend de escalado) sobre una incidencia concreta. El modelo **investiga
el cluster en solo-lectura** (`kubectl_toolbox`, allow-list get/describe/logs/top)
y **propone** un plan de remediación multi-paso — **nunca ejecuta**: la ejecución
va por el executor gateado (dry-run + aprobación por paso + L3 nunca), igual que
cualquier otro plan.

Un turno del usuario dispara un bucle acotado THOUGHT → ACTION (read-only) →
OBSERVATION, que termina en una respuesta en lenguaje natural y, opcionalmente,
un PLAN en JSON (mismo contrato/validación que el escalado single-shot).

Seguridad: el modelo solo dispone de la herramienta read-only. Aunque emita un
verbo de escritura, `kubectl_toolbox.execute` lo rechaza y el error vuelve como
observación — no se ejecuta nada. La validación del plan reutiliza
`escalation._parse_plan` (descarta destructivos/placeholders).
"""

from __future__ import annotations

import re

from src.diagnostics import escalation
from src.diagnostics.kubectl_toolbox import execute as kubectl_execute

# Historial por sesión (in-memory; se pierde al reiniciar — suficiente para F1).
_SESSIONS: dict[str, list[dict]] = {}
_MAX_HISTORY = 40          # mensajes conservados por sesión (poda los más viejos)
_MAX_ACTIONS_PER_TURN = 4  # tope de investigaciones read-only por mensaje
_MAX_OBS_CHARS = 1500

_ACTION_RE = re.compile(r"^\s*ACTION:\s*(kubectl .+)$", re.MULTILINE)
_PLAN_RE = re.compile(r"PLAN:\s*(\[.*\])", re.DOTALL)

_SYSTEM = """Eres un SRE experto en Kubernetes que ayuda a un operador a remediar
una incidencia, conversando en español. Puedes INVESTIGAR el cluster en SOLO
LECTURA y luego PROPONER un plan — nunca ejecutas acciones tú mismo.

En cada respuesta usa EXACTAMENTE UNO de estos formatos:

Formato A — investigar (UNA acción de solo lectura, un solo comando simple):
ACTION: kubectl <get|describe|logs|top> ... (usa nombres reales del namespace)
  · Un único comando por ACTION. NADA de `&&`, `|`, `;` ni redirecciones.
  · `kubectl logs` requiere un pod/`deploy/NAME` válido; si no lo conoces, primero `get pods`.

Formato B — responder al operador (y opcionalmente proponer un plan):
<respuesta en lenguaje natural, breve y concreta>
PLAN:
[
  {"type": "investigate"|"command", "action": "kubectl ...", "explanation": "...", "risk": 0|1|2}
]

Reglas del PLAN (solo si propones uno):
- "command" solo del conjunto seguro: `rollout restart`/`rollout undo`, `scale`,
  `set image`/`set resources`/`set env`. Usa nombres/valores reales observados.
- PROHIBIDO en el plan: delete, drain, cordon, exec, apply, create, patch. Si el
  arreglo necesita algo externo (crear secret, provisionar PV, añadir nodo), NO
  lo pongas como paso: dilo en la respuesta como acción externa.
- Si aún no tienes evidencia suficiente, investiga (Formato A) antes de proponer.
- No incluyas PLAN si el operador solo pregunta algo; responde y ya está."""


def _context(root_cause: str, evidence: str, namespace: str) -> str:
    ev = (evidence or "")[:1500]
    return (
        f"Incidencia — namespace: {namespace or '(desconocido)'}\n"
        f"Diagnóstico (causa raíz): {root_cause or '(sin diagnóstico)'}\n"
        f"Evidencia (eventos/logs):\n{ev}"
    )


def _history(session_id: str, root_cause: str, evidence: str, namespace: str) -> list[dict]:
    hist = _SESSIONS.get(session_id)
    if hist is None:
        hist = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _context(root_cause, evidence, namespace)},
            {"role": "assistant", "content": "Entendido. ¿Qué quieres que revise o remedie?"},
        ]
        _SESSIONS[session_id] = hist
    return hist


def _prune(hist: list[dict]) -> None:
    """Conserva el system + los últimos mensajes (evita crecer sin límite)."""
    if len(hist) <= _MAX_HISTORY:
        return
    system = hist[0]
    del hist[1:len(hist) - (_MAX_HISTORY - 1)]
    if hist[0] is not system:
        hist.insert(0, system)


def _run_readonly(action: str) -> str:
    """Ejecuta la acción de investigación (el toolbox rechaza escritura)."""
    res = kubectl_execute(action)
    out = getattr(res, "stdout", "") or getattr(res, "error", "") or "(sin salida)"
    return out[:_MAX_OBS_CHARS]


def _finalize(text: str, observations: list[dict], hist: list[dict]) -> dict:
    """Extrae plan validado + respuesta limpia de un texto Formato B."""
    proposed: list[dict] = []
    plan_m = _PLAN_RE.search(text or "")
    if plan_m:
        steps = escalation._parse_plan(plan_m.group(0))  # valida seguridad/riesgo
        proposed = [
            {"type": s.action_type, "action": s.action,
             "explanation": s.explanation, "risk": s.risk_level}
            for s in steps
        ]
    # Limpia el bloque PLAN y cualquier línea ACTION:/OBSERVATION: residual.
    reply = _PLAN_RE.sub("", text or "")
    reply = re.sub(r"^\s*(ACTION|OBSERVATION):.*$", "", reply, flags=re.MULTILINE).strip()
    if not reply:
        reply = "(He investigado; revisa el plan propuesto)" if proposed else \
                "(No tengo una conclusión clara; dame más detalle o pídeme que investigue algo concreto.)"
    _prune(hist)
    return {"reply": reply, "observations": observations, "proposed_plan": proposed}


def chat_turn(session_id: str, message: str, root_cause: str = "",
              evidence: str = "", namespace: str = "") -> dict:
    """Procesa un mensaje del operador. Devuelve:
        {reply, observations: [{action, output}], proposed_plan: [step dicts]}

    `proposed_plan` solo aparece si el modelo propuso uno válido (tras validación).
    """
    hist = _history(session_id, root_cause, evidence, namespace)
    hist.append({"role": "user", "content": message})
    observations: list[dict] = []
    seen: set[str] = set()

    for _ in range(_MAX_ACTIONS_PER_TURN):
        text = escalation.chat_completion(hist)
        if not text:
            hist.pop()  # deshaz el user para no envenenar el historial
            return {"reply": "", "observations": observations, "proposed_plan": [],
                    "error": "El modelo de chat no respondió (¿ESCALATION_BACKEND?)."}
        hist.append({"role": "assistant", "content": text})

        m = _ACTION_RE.search(text)
        plan_m = _PLAN_RE.search(text)
        # Investigación read-only: si hay ACTION y NO hay ya un PLAN final.
        if m and not plan_m:
            action = m.group(1).strip()
            if action in seen:
                break  # el modelo repite acción → deja de investigar y fuerza respuesta
            seen.add(action)
            obs = _run_readonly(action)
            observations.append({"action": action, "output": obs})
            hist.append({"role": "user",
                         "content": f"OBSERVATION:\n{obs}\n\nContinúa o responde al operador."})
            continue

        return _finalize(text, observations, hist)  # respuesta final del modelo

    # Presupuesto de investigación agotado (o acción repetida): fuerza respuesta final.
    hist.append({"role": "user", "content":
                 "Ya has investigado suficiente. Responde AHORA al operador (Formato B) con tu "
                 "conclusión y, si procede, un PLAN. No emitas más ACTION."})
    text = escalation.chat_completion(hist) or ""
    hist.append({"role": "assistant", "content": text})
    return _finalize(text, observations, hist)


def reset_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)
