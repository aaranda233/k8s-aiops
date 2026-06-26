"""Planner agéntico de remediación (escalado para problemas novedosos).

A diferencia del escalado *single-shot* de [escalation.py](escalation.py) —una
sola llamada ciega que devuelve un plan en JSON— este planner **investiga el
cluster en vivo** con kubectl de solo lectura (ciclo ReAct: THOUGHT → ACTION →
OBSERVATION) y, cuando tiene evidencia suficiente, emite un PLAN multi-paso.

Reutiliza la maquinaria existente:
  - `kubectl_toolbox.execute` para las observaciones (allowlist read-only).
  - `escalation._parse_plan` para parsear y VALIDAR el plan final (solo verbos de
    lectura + `rollout restart`; cualquier verbo destructivo se descarta).

El plan resultante (`list[Step]`) lo persiste `resolve_with_escalation` como
provisional en el grafo (`add_provisional`), igual que el camino single-shot, así
que todo aguas abajo (UI, ejecución paso a paso, verificación por outcome,
consolidación ORPO) funciona sin cambios.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace

import httpx

from src.diagnostics.escalation import _parse_plan as parse_plan
from src.diagnostics.kubectl_toolbox import execute as kubectl_execute
from src.remediation.remediation_graph import COMMAND, INVESTIGATE, Step

_DEFAULT_MODEL = "qwen2.5-coder:14b"
_MAX_OBS_CHARS = 1500

# Placeholder sin resolver (p. ej. <pod>, <nombre-del-pod>, <deployment>). Un paso
# ejecutable que lo conserve no es accionable → se descarta.
_UNRESOLVED_RE = re.compile(r"<[^>\n]+>")

_SYSTEM = """You are an expert Kubernetes SRE. A monitoring system detected an
anomaly and hands you a diagnosis plus event evidence. First INVESTIGATE the live
cluster with read-only kubectl to learn the REAL resource names, then produce a
step-by-step remediation PLAN that uses those exact names.

Each turn output EXACTLY ONE of these two formats:

Format A — investigate further:
THOUGHT: reasoning about what to check next
ACTION: kubectl <read-only command>

Format B — final plan (only when you have enough evidence):
THOUGHT: summary of what you found
PLAN:
[
  {"type": "investigate"|"command", "action": "...", "explanation": "...", "risk": 0|1|2}
]

How to investigate (Format A) — read only:
- ALWAYS start by listing resources to discover their real names, e.g.
  `kubectl get pods -n aiops-demo`, then `kubectl describe pod <that-name> -n aiops-demo`.
- Read the exact pod / deployment / service names AND current values (image tag,
  memory/cpu limits, env) from the OBSERVATION output — you need them for the fix.
- Read verbs allowed: get, describe, logs, top, events.

PLAN rules (Format B) — every step must be EXECUTABLE, no manual steps:
- 2 to 5 steps, ordered: investigate -> identify -> fix -> verify.
- Use the EXACT resource names and real values you observed (e.g.
  `deployment/inventory-api`, `--limits=memory=512Mi`, `app=repo/img:1.2.3`).
  NEVER write a placeholder such as <pod>, <deployment>, <name>, <tag> or <x>.
  A step containing angle brackets is INVALID — investigate to get the value first.
- "investigate" -> ONE read-only kubectl line.
- "command" -> ONE kubectl line from this SAFE set only:
  `rollout restart` / `rollout undo` / `scale` / `set image` / `set resources` / `set env`.
- FORBIDDEN anywhere: delete, drain, cordon, exec, apply, create, patch, annotate, label.
- If the real fix needs an external action you CANNOT express as one of the safe
  commands above (create a secret with an unknown value, provision a PV, add node
  capacity), DO NOT invent a step — return only the investigation steps.
- "explanation" -> Spanish, brief.
- Leave the action command for the end.
- Output ONLY format A or B, nothing else."""


def _context(root_cause: str, evidence: str, namespace: str) -> str:
    ev = (evidence or "")[:1500]
    return (
        f"Namespace: {namespace}\n"
        f"Causa raíz (diagnóstico): {root_cause}\n"
        f"Evidencia (eventos):\n{ev}\n\n"
        "Investiga con kubectl de solo lectura y luego entrega el PLAN final."
    )


def _extract_action(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("ACTION:"):
            cmd = line.removeprefix("ACTION:").strip()
            return cmd or None
    return None


def _has_plan(text: str) -> bool:
    """True si el texto contiene un array JSON de objetos (el PLAN final)."""
    return re.search(r"\[\s*\{", text or "") is not None


def _resolve_steps(steps: list[Step]) -> list[Step]:
    """Descarta los pasos ejecutables que aún traen un placeholder `<...>` sin
    resolver y reindexa el orden. La guía manual (texto) se conserva."""
    kept = [
        s for s in steps
        if not (s.action_type in (INVESTIGATE, COMMAND) and _UNRESOLVED_RE.search(s.action))
    ]
    return [replace(s, order=i) for i, s in enumerate(kept)]


class AgenticPlanner:
    """Ciclo investiga→plan contra un modelo grande local (Ollama)."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        max_steps: int = 4,
        timeout: float = 180.0,
        tool=kubectl_execute,
    ):
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("ESCALATION_MODEL", "") or _DEFAULT_MODEL
        self.max_steps = max_steps
        self.timeout = timeout
        self.tool = tool

    def plan(self, root_cause: str, evidence: str, namespace: str) -> list[Step]:
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _context(root_cause, evidence, namespace)},
        ]
        seen_actions: set[str] = set()

        for _ in range(self.max_steps):
            text = self._call_llm(messages)
            if _has_plan(text):
                steps = _resolve_steps(parse_plan(text))
                if steps:
                    return steps
            action = _extract_action(text)
            if not action or action in seen_actions:
                break
            seen_actions.add(action)
            observation = self._run_tool(action)
            messages.append({"role": "assistant", "content": text})
            messages.append({
                "role": "user",
                "content": f"OBSERVATION:\n{observation}\n\n"
                           "Continúa investigando o entrega el PLAN final en JSON.",
            })

        # Límite alcanzado o sin nueva acción: fuerza el plan final.
        messages.append({
            "role": "user",
            "content": "Entrega AHORA el PLAN final como array JSON (Format B) usando los "
                       "nombres reales observados, sin placeholders ni texto alrededor.",
        })
        return _resolve_steps(parse_plan(self._call_llm(messages)))

    def _call_llm(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 700},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _run_tool(self, action: str) -> str:
        """Ejecuta la acción de investigación (read-only) y devuelve la observación.

        El toolbox rechaza cualquier verbo de escritura, así que un comando
        peligroso del modelo vuelve como error en vez de ejecutarse.
        """
        res = self.tool(action)
        out = getattr(res, "stdout", "") or getattr(res, "error", "") or "(sin salida)"
        return out[:_MAX_OBS_CHARS]

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as c:
                resp = c.get(f"{self.host}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                return any(self.model in m for m in models)
        except Exception:
            return False


def plan_agentic(root_cause: str, evidence: str, namespace: str) -> list[Step]:
    """Entrada best-effort: devuelve Steps validados o [] si el escalado falla.

    Devolver [] permite que `escalate()` caiga al camino single-shot.
    """
    try:
        return AgenticPlanner().plan(root_cause, evidence, namespace)
    except Exception:
        return []
