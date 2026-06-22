"""Tests del escalado a modelo grande (src/diagnostics/escalation.py).

Verifica el parser de planes, la validación de seguridad (solo lectura +
rollout restart; rechaza destructivos) y que está desactivado por defecto.
No requiere API ni Ollama (no se llama a ningún backend).
"""

import pytest

from src.diagnostics import escalation
from src.remediation.remediation_graph import COMMAND, GUIDANCE, INVESTIGATE


@pytest.mark.unit
def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ESCALATION_BACKEND", raising=False)
    assert escalation.is_enabled() is False


@pytest.mark.unit
def test_command_safety():
    assert escalation._command_is_safe("kubectl get pods -n x")
    assert escalation._command_is_safe("kubectl describe ingress -n x")
    assert escalation._command_is_safe("kubectl rollout restart deployment/a -n x")
    assert not escalation._command_is_safe("kubectl delete pod a -n x")
    assert not escalation._command_is_safe("kubectl apply -f x.yaml")
    assert not escalation._command_is_safe("rm -rf /")


@pytest.mark.unit
def test_parse_plan_valid():
    text = """Aquí tienes el plan:
    [
      {"type":"investigate","action":"kubectl get endpoints -n web","explanation":"ver backend","risk":0},
      {"type":"guidance","action":"Corrige el Ingress","explanation":"","risk":0},
      {"type":"command","action":"kubectl rollout restart deployment/web -n web","explanation":"reinicia","risk":1}
    ]"""
    steps = escalation._parse_plan(text)
    assert len(steps) == 3
    assert steps[0].action_type == INVESTIGATE
    assert steps[1].action_type == GUIDANCE
    assert steps[2].action_type == COMMAND
    assert steps[2].risk_level == 1
    assert all(s.source == "escalated" and not s.verified for s in steps)


@pytest.mark.unit
def test_parse_plan_drops_destructive():
    text = """[
      {"type":"command","action":"kubectl delete pod web-0 -n web","explanation":"borra","risk":3},
      {"type":"investigate","action":"kubectl get pods -n web","explanation":"lista","risk":0}
    ]"""
    steps = escalation._parse_plan(text)
    # el delete (destructivo) se descarta; solo queda el get
    assert len(steps) == 1
    assert steps[0].action == "kubectl get pods -n web"


@pytest.mark.unit
def test_parse_plan_garbage_returns_empty():
    assert escalation._parse_plan("no hay json aquí") == []
    assert escalation._parse_plan("") == []


@pytest.mark.unit
def test_escalate_returns_empty_when_disabled(monkeypatch):
    monkeypatch.delenv("ESCALATION_BACKEND", raising=False)
    assert escalation.escalate("rc", "ev", "ns") == []
