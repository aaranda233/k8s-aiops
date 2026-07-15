"""Tests del chat de remediación (src/diagnostics/remediation_chat.py).

Verifica: el bucle investiga en solo-lectura y NUNCA ejecuta escritura, el plan
propuesto se valida (destructivos descartados), y el caso sin backend. El LLM y
el toolbox se simulan — no requiere modelo ni cluster.
"""

import pytest

from src.diagnostics import escalation, remediation_chat


class _ToolResult:
    def __init__(self, stdout="", error=""):
        self.stdout = stdout
        self.error = error


@pytest.fixture(autouse=True)
def _clean_sessions():
    remediation_chat._SESSIONS.clear()
    yield
    remediation_chat._SESSIONS.clear()


@pytest.mark.unit
def test_investigate_then_answer_no_writes(monkeypatch):
    """El modelo investiga (ACTION read-only) y luego responde sin plan; solo se
    ejecuta el comando de lectura y nada más."""
    turns = iter([
        "ACTION: kubectl get pods -n web",
        "Veo un pod en CrashLoop; revisa la config.",
    ])
    monkeypatch.setattr(escalation, "chat_completion", lambda m, timeout=120.0: next(turns))
    executed = []
    monkeypatch.setattr(remediation_chat, "kubectl_execute",
                        lambda cmd: executed.append(cmd) or _ToolResult(stdout="api-1 CrashLoop"))

    out = remediation_chat.chat_turn("s1", "qué pasa", "CrashLoop", "events", "web")
    assert executed == ["kubectl get pods -n web"]
    assert out["observations"][0]["action"] == "kubectl get pods -n web"
    assert out["proposed_plan"] == []
    assert "CrashLoop" in out["reply"]


@pytest.mark.unit
def test_proposed_plan_drops_destructive(monkeypatch):
    """Un plan con un paso destructivo y otro seguro conserva solo el seguro."""
    plan_json = ('[{"type":"command","action":"kubectl delete pod x -n web","risk":3},'
                 '{"type":"command","action":"kubectl rollout restart deployment/api -n web","risk":1}]')
    turns = iter([f"Propongo esto.\nPLAN:\n{plan_json}"])
    monkeypatch.setattr(escalation, "chat_completion", lambda m, timeout=120.0: next(turns))
    monkeypatch.setattr(remediation_chat, "kubectl_execute", lambda cmd: _ToolResult(stdout=""))

    out = remediation_chat.chat_turn("s2", "arréglalo", "CrashLoop", "ev", "web")
    actions = [s["action"] for s in out["proposed_plan"]]
    assert actions == ["kubectl rollout restart deployment/api -n web"]
    assert "PLAN:" not in out["reply"]  # el bloque JSON se recorta de la respuesta


@pytest.mark.unit
def test_write_action_is_not_executed_by_model(monkeypatch):
    """Si el modelo intenta una ACCIÓN de escritura, el toolbox la rechaza: el
    chat nunca ejecuta escritura (defensa del toolbox real, sin monkeypatch)."""
    turns = iter([
        "ACTION: kubectl delete deployment api -n web",
        "No pude, es de solo lectura.",
    ])
    monkeypatch.setattr(escalation, "chat_completion", lambda m, timeout=120.0: next(turns))
    # Usa el toolbox REAL: 'delete' es verbo prohibido → devuelve error sin ejecutar.
    out = remediation_chat.chat_turn("s3", "borra el deployment", "x", "y", "web")
    obs = out["observations"][0]
    assert "delete" in obs["action"]
    assert "solo lectura" in obs["output"].lower() or "prohibido" in obs["output"].lower()


@pytest.mark.unit
def test_no_backend_returns_error(monkeypatch):
    monkeypatch.setattr(escalation, "chat_completion", lambda m, timeout=120.0: None)
    out = remediation_chat.chat_turn("s4", "hola", "x", "y", "web")
    assert out["reply"] == "" and out["proposed_plan"] == []
    assert "error" in out
