"""Escalado a un modelo grande cuando el grafo de remediación no tiene plan (miss).

Pluggable por variables de entorno y **desactivado por defecto** (cero cambio en
producción hasta que se configure):

  ESCALATION_BACKEND = none | anthropic | openai | ollama   (default: none)
  ESCALATION_MODEL   = <nombre del modelo>
  ANTHROPIC_API_KEY / OPENAI_API_KEY   (según backend)
  OLLAMA_HOST                          (backend ollama, default localhost:11434)

El modelo grande propone un PLAN multi-paso en JSON estricto; se parsea a Steps
y se VALIDA: solo verbos de lectura (get/describe/logs/top/events) + la acción
reversible `rollout restart`. Cualquier verbo destructivo (delete/drain/cordon/
exec/apply/patch...) se rechaza — coherente con el modo shadow del sistema.

Devuelve None si está desactivado, falla la llamada, o el plan no valida.
"""

from __future__ import annotations

import json
import os
import re

import httpx

from src.remediation.remediation_graph import COMMAND, GUIDANCE, INVESTIGATE, Step

_READ_VERBS = {"get", "describe", "logs", "top", "events"}
_VALID_TYPES = {INVESTIGATE, COMMAND, GUIDANCE}

_SYSTEM = """Eres un SRE experto en Kubernetes. Te doy un problema anómalo y debes
proponer un PLAN de remediación MULTI-PASO (investigar → identificar → arreglar →
verificar).

Reglas estrictas:
- Responde SOLO con un array JSON, sin texto alrededor.
- Cada paso: {"type": "investigate"|"command"|"guidance", "action": "...", "explanation": "...", "risk": 0|1}
- "command"/"investigate" → un comando kubectl en UNA línea. SOLO verbos de lectura
  (get/describe/logs/top/events) o `kubectl rollout restart deployment/<x> -n <ns>`.
- PROHIBIDO delete, drain, cordon, exec, apply, patch, scale u otros destructivos.
- "guidance" → texto en español de una acción manual (sin comando).
- explanation: español, breve.
- 2 a 5 pasos. Empieza investigando, deja la acción reversible para el final."""


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


def _call_backend(system: str, user: str) -> str | None:
    """Llama al backend configurado. Devuelve el texto crudo o None."""
    backend = (os.getenv("ESCALATION_BACKEND", "none") or "none").lower()
    model = os.getenv("ESCALATION_MODEL", "")
    try:
        if backend == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY", "")
            if not key:
                return None
            r = httpx.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json={"model": model or "claude-sonnet-4-6", "max_tokens": 700,
                      "system": system, "messages": [{"role": "user", "content": user}]},
                timeout=60.0,
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]
        if backend == "openai":
            key = os.getenv("OPENAI_API_KEY", "")
            if not key:
                return None
            r = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model or "gpt-4o", "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}]},
                timeout=60.0,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        if backend == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            r = httpx.post(
                f"{host}/api/chat",
                json={"model": model or "qwen2.5:1.5b", "stream": False, "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}]},
                timeout=120.0,
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
    except Exception:
        return None
    return None


def _command_is_safe(action: str) -> bool:
    parts = action.split()
    if len(parts) < 2 or parts[0] != "kubectl":
        return False
    if "rollout restart" in action:
        return True
    verb = parts[1].lower()
    return verb in _READ_VERBS


def _parse_plan(text: str) -> list[Step]:
    """Extrae el array JSON y construye Steps validados (descarta inseguros)."""
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
        if atype in (INVESTIGATE, COMMAND):
            if not _command_is_safe(action):
                continue  # comando inseguro/destructivo → descartar
            risk = 1 if "rollout restart" in action else 0
        else:
            risk = 0
        steps.append(Step(
            order=order, action_type=atype, action=action,
            explanation=str(item.get("explanation", "")).strip(),
            risk_level=risk, source="escalated", verified=False,
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


def resolve_with_escalation(evidence: str, namespace: str, root_cause: str = ""):
    """Punto de integración único para el pipeline.

    Intenta el grafo (catálogo + embedding); si es miss y el escalado está
    habilitado, pide un plan al modelo grande, lo persiste como provisional
    (sin verificar) para reusarlo en futuros miss, y lo devuelve. Si todo falla,
    None → el pipeline cae a su remediación determinista actual.
    """
    import hashlib

    from src.remediation.remediation_graph import Plan, get_graph

    g = get_graph()
    plan = g.resolve(evidence, namespace, root_cause)
    if plan is not None:
        return plan
    if not is_enabled():
        return None
    steps = escalate(root_cause, evidence, namespace)
    if not steps:
        return None
    key = "escalated:" + hashlib.md5((root_cause or "")[:200].encode()).hexdigest()[:10]
    g.add_provisional(key, steps, signature_text=f"{root_cause}\n{evidence[:400]}")
    return Plan(intent=key, namespace=namespace, steps=steps, source="escalated")
