"""Tests del planner agéntico (src/diagnostics/agentic_planner.py).

Verifica el ciclo investiga→plan con el LLM y el toolbox mockeados (sin Ollama
ni cluster): que investiga antes de planificar, que el plan final se parsea y
valida (rechaza destructivos, reutiliza el validador de escalation), que respeta
el límite de pasos y que el wrapper best-effort devuelve [] ante fallos.
"""

from dataclasses import dataclass

import pytest

from src.diagnostics import agentic_planner
from src.diagnostics.agentic_planner import AgenticPlanner, plan_agentic
from src.remediation.remediation_graph import COMMAND, INVESTIGATE


@dataclass
class _FakeToolResult:
    command: str
    stdout: str = ""
    returncode: int = 0
    error: str | None = None


def _scripted_llm(responses):
    """Devuelve una función _call_llm que va soltando respuestas en orden."""
    seq = list(responses)

    def _call(self, messages):
        return seq.pop(0) if seq else "PLAN:\n[]"

    return _call


_VALID_PLAN = """THOUGHT: ya tengo evidencia
PLAN:
[
  {"type":"investigate","action":"kubectl get endpoints -n web","explanation":"ver backend","risk":0},
  {"type":"command","action":"kubectl rollout restart deployment/web -n web","explanation":"reinicia","risk":1}
]"""


@pytest.mark.unit
def test_investigates_then_plans(monkeypatch):
    """Primero emite una ACTION read-only, observa, y luego entrega el PLAN."""
    calls = []
    monkeypatch.setattr(
        AgenticPlanner, "_call_llm",
        _scripted_llm([
            "THOUGHT: reviso endpoints\nACTION: kubectl get endpoints -n web",
            _VALID_PLAN,
        ]),
    )

    def fake_tool(cmd):
        calls.append(cmd)
        return _FakeToolResult(command=cmd, stdout="NAME ENDPOINTS\nweb <none>")

    p = AgenticPlanner(tool=fake_tool, max_steps=4)
    steps = p.plan("backend sin endpoints", "evidencia", "web")

    assert calls == ["kubectl get endpoints -n web"]  # investigó en vivo
    assert len(steps) == 2
    assert steps[0].action_type == INVESTIGATE
    assert steps[1].action_type == COMMAND
    assert "rollout restart" in steps[1].action


@pytest.mark.unit
def test_plan_rejects_destructive_commands(monkeypatch):
    """Un paso destructivo en el plan final se descarta (validador de escalation)."""
    monkeypatch.setattr(
        AgenticPlanner, "_call_llm",
        _scripted_llm(["""PLAN:
[
  {"type":"investigate","action":"kubectl get pods -n web","explanation":"ok","risk":0},
  {"type":"command","action":"kubectl delete deployment/web -n web","explanation":"malo","risk":1}
]"""]),
    )
    p = AgenticPlanner(tool=lambda c: _FakeToolResult(command=c), max_steps=2)
    steps = p.plan("rc", "ev", "web")
    actions = [s.action for s in steps]
    assert "kubectl get pods -n web" in actions
    assert all("delete" not in a for a in actions)


@pytest.mark.unit
def test_forces_final_plan_when_step_limit_reached(monkeypatch):
    """Si solo investiga hasta el límite, se le fuerza el PLAN final."""
    monkeypatch.setattr(
        AgenticPlanner, "_call_llm",
        _scripted_llm([
            "THOUGHT: a\nACTION: kubectl get pods -n web",
            "THOUGHT: b\nACTION: kubectl describe pod x -n web",
            _VALID_PLAN,  # respuesta a la petición forzada
        ]),
    )
    p = AgenticPlanner(tool=lambda c: _FakeToolResult(command=c, stdout="ok"), max_steps=2)
    steps = p.plan("rc", "ev", "web")
    assert len(steps) == 2  # del PLAN forzado


@pytest.mark.unit
def test_repeated_action_breaks_loop(monkeypatch):
    """Si el modelo repite la misma ACTION, no entra en bucle: fuerza el final."""
    monkeypatch.setattr(
        AgenticPlanner, "_call_llm",
        _scripted_llm([
            "ACTION: kubectl get pods -n web",
            "ACTION: kubectl get pods -n web",  # repetida → rompe
            _VALID_PLAN,
        ]),
    )
    calls = []
    p = AgenticPlanner(tool=lambda c: calls.append(c) or _FakeToolResult(command=c, stdout="ok"),
                       max_steps=5)
    steps = p.plan("rc", "ev", "web")
    assert calls == ["kubectl get pods -n web"]  # solo se ejecutó una vez
    assert len(steps) == 2


@pytest.mark.unit
def test_plan_agentic_returns_empty_on_failure(monkeypatch):
    """El wrapper best-effort devuelve [] si el planner lanza (p. ej. Ollama caído)."""
    def boom(self, *a, **k):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(AgenticPlanner, "plan", boom)
    assert plan_agentic("rc", "ev", "web") == []


@pytest.mark.unit
def test_drops_placeholders_and_guidance(monkeypatch):
    """Se descarta el paso con `<...>` sin resolver Y el paso manual (guidance)."""
    monkeypatch.setattr(
        AgenticPlanner, "_call_llm",
        _scripted_llm(["""PLAN:
[
  {"type":"investigate","action":"kubectl describe pod <nombre-del-pod> -n web","explanation":"placeholder","risk":0},
  {"type":"investigate","action":"kubectl get pods -n web","explanation":"real","risk":0},
  {"type":"guidance","action":"Revisa la config del deployment","explanation":"manual","risk":0}
]"""]),
    )
    p = AgenticPlanner(tool=lambda c: _FakeToolResult(command=c), max_steps=1)
    steps = p.plan("rc", "ev", "web")
    actions = [s.action for s in steps]
    assert actions == ["kubectl get pods -n web"]  # solo sobrevive el paso real
    assert all(s.action_type != "guidance" for s in steps)  # sin pasos manuales
    assert [s.order for s in steps] == list(range(len(steps)))  # reindexado


@pytest.mark.unit
def test_extract_action_and_has_plan():
    assert agentic_planner._extract_action("THOUGHT: x\nACTION: kubectl get pods") == "kubectl get pods"
    assert agentic_planner._extract_action("THOUGHT: solo pienso") is None
    assert agentic_planner._has_plan('PLAN:\n[ {"type":"investigate"} ]') is True
    assert agentic_planner._has_plan("THOUGHT: nada de json") is False


@pytest.mark.unit
def test_escalate_dispatches_to_agentic(monkeypatch):
    """escalate() con ESCALATION_MODE=agentic usa el planner agéntico."""
    from src.diagnostics import escalation

    monkeypatch.setenv("ESCALATION_BACKEND", "ollama")
    monkeypatch.setenv("ESCALATION_MODE", "agentic")
    sentinel = ["USED_AGENTIC"]
    monkeypatch.setattr(
        "src.diagnostics.agentic_planner.plan_agentic",
        lambda rc, ev, ns: [__import__("src.remediation.remediation_graph", fromlist=["Step"]).Step(
            order=0, action_type=INVESTIGATE, action="kubectl get pods -n web")],
    )
    steps = escalation.escalate("rc", "ev", "web")
    assert len(steps) == 1
    assert steps[0].action == "kubectl get pods -n web"
