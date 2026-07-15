"""Escalado a un modelo grande cuando el grafo de remediación no resuelve con un
comando ejecutable (miss, o plan sin paso de acción).

Pluggable por variables de entorno y **desactivado por defecto** (cero cambio en
producción hasta que se configure):

  ESCALATION_BACKEND = none | anthropic | openai | ollama   (default: none)
  ESCALATION_MODEL   = <nombre del modelo>
  ESCALATION_BASE_URL= base URL OpenAI-compatible (backend openai;
                       default https://api.openai.com/v1). Apunta aquí un
                       endpoint local tipo vLLM (p. ej. un Qwen grande en otra
                       máquina) para sustituir el modelo de escalado sin egress.
  ANTHROPIC_API_KEY / OPENAI_API_KEY   (según backend; la key es OPCIONAL cuando
                       ESCALATION_BASE_URL apunta a un endpoint local sin auth)
  OLLAMA_HOST                          (backend ollama, default localhost:11434)

El modelo grande propone un PLAN multi-paso; se parsea a Steps y se VALIDA contra
un vocabulario seguro: verbos de lectura (get/describe/logs/top/events) + acciones
reversibles/de-configuración (rollout restart/undo, scale, set image/resources/env).
Cualquier verbo destructivo (delete/drain/cordon/exec/apply/create/patch...) se
rechaza. **No existen pasos manuales** (`guidance`): cada paso es ejecutable.

Devuelve None/[] si está desactivado, falla la llamada, o el plan no valida.
"""

from __future__ import annotations

import json
import os
import re

import httpx

from src.remediation.remediation_graph import COMMAND, INVESTIGATE, Step
from src.remediation.risk_scorer import score

_READ_VERBS = {"get", "describe", "logs", "top", "events"}
# Acciones de escritura permitidas (reversibles L1 + configuración L2). Cada una
# pasa por dry-run + aprobación por paso en el executor. Destructivos prohibidos.
_SAFE_WRITE_PREFIXES = [
    ("rollout", "restart"), ("rollout", "undo"), ("scale",),
    ("set", "image"), ("set", "resources"), ("set", "env"),
]
_VALID_TYPES = {INVESTIGATE, COMMAND}

_SYSTEM = """Eres un SRE experto en Kubernetes. Te doy un problema anómalo y debes
proponer un PLAN de remediación MULTI-PASO (investigar → identificar → arreglar →
verificar) totalmente EJECUTABLE — sin pasos manuales.

Reglas estrictas:
- Responde SOLO con un array JSON, sin texto alrededor.
- Cada paso: {"type": "investigate"|"command", "action": "...", "explanation": "...", "risk": 0|1|2}
- "investigate" → comando kubectl de SOLO LECTURA (get/describe/logs/top/events).
- "command" → comando kubectl de acción, UNA línea, SOLO de este conjunto seguro:
  `rollout restart` / `rollout undo` / `scale` / `set image` / `set resources` / `set env`.
  Usa nombres y valores reales (lee el recurso antes para conocer imagen/límites actuales).
- PROHIBIDO: delete, drain, cordon, exec, apply, create, patch, annotate, label.
- NO uses pasos de texto/manuales: si el arreglo necesita una acción externa
  (crear un secret, provisionar un PV, añadir un nodo), NO lo incluyas en el plan;
  limítate a los pasos de investigación que sí puedes ejecutar.
- explanation: español, breve.
- 2 a 5 pasos. Empieza investigando, deja la acción al final."""


def is_enabled() -> bool:
    return (os.getenv("ESCALATION_BACKEND", "none") or "none").lower() != "none"


def _user_prompt(root_cause: str, evidence: str, namespace: str) -> str:
    ev = (evidence or "")[:1500]
    return (
        f"Namespace: {namespace}\n"
        f"Causa raíz (diagnóstico): {root_cause}\n"
        f"Evidencia (eventos):\n{ev}\n\n"
        "Devuelve SOLO el array JSON del plan."
    )


def chat_completion(messages: list[dict], timeout: float = 120.0) -> str | None:
    """Completa un chat multi-turno con el backend configurado (mismo que el
    escalado). `messages` = [{"role","content"}, …]. Devuelve el texto o None si
    el backend está off / falla / la API pública exige key ausente.

    Un endpoint OpenAI-compatible local (p. ej. el Qwen grande del GB10 vía
    ESCALATION_BASE_URL) no requiere key.
    """
    backend = (os.getenv("ESCALATION_BACKEND", "none") or "none").lower()
    model = os.getenv("ESCALATION_MODEL", "")
    try:
        if backend == "openai":
            base = (os.getenv("ESCALATION_BASE_URL", "") or "https://api.openai.com/v1").rstrip("/")
            key = os.getenv("OPENAI_API_KEY", "")
            if "api.openai.com" in base and not key:
                return None
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            r = httpx.post(
                f"{base}/chat/completions", headers=headers,
                json={"model": model or "gpt-4o", "messages": messages},
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        if backend == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            r = httpx.post(
                f"{host}/api/chat",
                json={"model": model or "qwen2.5:1.5b", "stream": False, "messages": messages},
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        if backend == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY", "")
            if not key:
                return None
            system = " ".join(m["content"] for m in messages if m.get("role") == "system")
            conv = [m for m in messages if m.get("role") != "system"]
            r = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json={"model": model or "claude-sonnet-4-6", "max_tokens": 900,
                      "system": system, "messages": conv},
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]
    except Exception:
        return None
    return None


def _call_backend(system: str, user: str) -> str | None:
    """Llamada single-shot (system+user) al backend configurado. Delega en
    `chat_completion` para no duplicar el cliente."""
    return chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )


def _command_is_safe(action: str) -> bool:
    """True si es un kubectl de lectura o del conjunto seguro de escritura.

    Rechaza placeholders sin resolver y cualquier verbo fuera del vocabulario.
    """
    parts = action.split()
    if len(parts) < 2 or parts[0] != "kubectl":
        return False
    if "<" in action and ">" in action:
        return False  # placeholder sin resolver
    verb = parts[1].lower()
    if verb in _READ_VERBS:
        return True
    rest = [p.lower() for p in parts[1:]]
    return any(rest[:len(p)] == list(p) for p in _SAFE_WRITE_PREFIXES)


def _parse_plan(text: str) -> list[Step]:
    """Extrae el array JSON y construye Steps validados (descarta inseguros).

    Solo investigate/command; el riesgo se calcula con el risk_scorer real. Los
    items `guidance` (manuales) u otros tipos se descartan."""
    if not text:
        return []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        raw = json.loads(m.group(0))
    except (ValueError, TypeError):
        return []
    steps: list[Step] = []
    order = 0
    for item in raw:
        if not isinstance(item, dict):
            continue
        atype = str(item.get("type", "")).strip().lower()
        action = str(item.get("action", "")).strip()
        if atype not in _VALID_TYPES or not action:
            continue
        if not _command_is_safe(action):
            continue  # comando inseguro/destructivo/placeholder → descartar
        steps.append(Step(
            order=order, action_type=atype, action=action,
            explanation=str(item.get("explanation", "")).strip(),
            risk_level=score(action).level, source="escalated", verified=False,
        ))
        order += 1
    return steps


def escalate(root_cause: str, evidence: str, namespace: str) -> list[Step]:
    """Pide un plan al modelo grande y devuelve Steps validados ([] si falla/off).

    ESCALATION_MODE=agentic → el planner investiga el cluster en vivo (read-only)
    antes de proponer el plan; si no produce nada, cae al single-shot. Por defecto
    (single_shot) hace una sola llamada ciega.
    """
    if not is_enabled():
        return []
    mode = (os.getenv("ESCALATION_MODE", "single_shot") or "single_shot").lower()
    if mode == "agentic":
        from src.diagnostics.agentic_planner import plan_agentic

        steps = plan_agentic(root_cause, evidence, namespace)
        if steps:
            return steps
    text = _call_backend(_SYSTEM, _user_prompt(root_cause, evidence, namespace))
    return _parse_plan(text or "")


def _has_command(steps) -> bool:
    """True si el plan tiene al menos un paso de acción ejecutable (no solo lectura)."""
    return any(getattr(s, "action_type", "") == COMMAND for s in (steps or []))


def resolve_with_escalation(evidence: str, namespace: str, root_cause: str = ""):
    """Punto de integración único para el pipeline.

    Intenta el grafo (catálogo + embedding). Escala al modelo grande cuando hay
    miss **o** cuando el plan del catálogo no tiene un paso de acción ejecutable
    (antes terminaba en un paso manual): el modelo investiga en vivo y propone un
    comando concreto, que se persiste como provisional para reusarlo. Si el modelo
    tampoco encuentra una acción segura, se devuelve el plan de investigación del
    catálogo (la nota de "acción externa" la aporta la capa determinista).
    """
    import hashlib

    from src.remediation.remediation_graph import Plan, get_graph

    g = get_graph()
    plan = g.resolve(evidence, namespace, root_cause)
    if plan is not None and _has_command(plan.steps):
        return plan
    if not is_enabled():
        return plan  # plan de investigación del catálogo (sin acción) o None
    steps = escalate(root_cause, evidence, namespace)
    if not _has_command(steps):
        return plan  # el modelo no halló acción segura → cae al catálogo / None
    key = "escalated:" + hashlib.md5((root_cause or "")[:200].encode()).hexdigest()[:10]
    g.add_provisional(key, steps, signature_text=f"{root_cause}\n{evidence[:400]}")
    return Plan(intent=key, namespace=namespace, steps=steps, source="escalated")
