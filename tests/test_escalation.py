"""Tests del escalado a modelo grande (src/diagnostics/escalation.py).

Verifica el parser de planes, la validación de seguridad (solo lectura +
rollout restart; rechaza destructivos) y que está desactivado por defecto.
No requiere API ni Ollama (no se llama a ningún backend).
"""

import pytest

from src.diagnostics import escalation
from src.remediation.remediation_graph import COMMAND, INVESTIGATE


@pytest.mark.unit
def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("ESCALATION_BACKEND", raising=False)
    assert escalation.is_enabled() is False


@pytest.mark.unit
def test_command_safety():
    # lectura
    assert escalation._command_is_safe("kubectl get pods -n x")
    assert escalation._command_is_safe("kubectl describe ingress -n x")
    # escritura segura (reversible + config)
    assert escalation._command_is_safe("kubectl rollout restart deployment/a -n x")
    assert escalation._command_is_safe("kubectl rollout undo deployment/a -n x")
    assert escalation._command_is_safe("kubectl scale deployment/a --replicas=2 -n x")
    assert escalation._command_is_safe("kubectl set image deployment/a c=repo/img:1.2 -n x")
    assert escalation._command_is_safe("kubectl set resources deployment/a --limits=memory=512Mi -n x")
    assert escalation._command_is_safe("kubectl set env deployment/a KEY=val -n x")
    # destructivo / fuera de vocabulario / placeholder
    assert not escalation._command_is_safe("kubectl delete pod a -n x")
    assert not escalation._command_is_safe("kubectl apply -f x.yaml")
    assert not escalation._command_is_safe("kubectl create secret generic s -n x")
    assert not escalation._command_is_safe("kubectl patch deployment/a -p '{}' -n x")
    assert not escalation._command_is_safe("kubectl set image deployment/a c=<img> -n x")  # placeholder
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
    # el paso guidance (manual) se descarta: solo sobreviven investigate + command
    assert len(steps) == 2
    assert steps[0].action_type == INVESTIGATE
    assert steps[1].action_type == COMMAND
    assert steps[1].risk_level == 1  # rollout restart → L1 (risk_scorer)
    assert [s.order for s in steps] == [0, 1]  # reindexado tras descartar el manual
    assert all(s.source == "escalated" and not s.verified for s in steps)


@pytest.mark.unit
def test_parse_plan_scores_config_write_as_l2():
    text = """[
      {"type":"command","action":"kubectl set resources deployment/a --limits=memory=512Mi -n x","explanation":"sube memoria"}
    ]"""
    steps = escalation._parse_plan(text)
    assert len(steps) == 1
    assert steps[0].risk_level == 2  # set resources → L2 (configuración)


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


@pytest.mark.unit
def test_has_command():
    from src.remediation.remediation_graph import Step
    inv = Step(order=0, action_type=INVESTIGATE, action="kubectl get pods -n x")
    cmd = Step(order=1, action_type=COMMAND, action="kubectl rollout restart deployment/a -n x")
    assert escalation._has_command([inv, cmd]) is True
    assert escalation._has_command([inv]) is False
    assert escalation._has_command([]) is False


@pytest.mark.unit
def test_resolve_escalates_when_plan_has_no_command(monkeypatch):
    """Si el catálogo solo da investigación (sin acción), se escala al modelo."""
    from src.remediation.remediation_graph import Plan, Step

    invest_only = Plan(intent="image", namespace="ns",
                       steps=[Step(order=0, action_type=INVESTIGATE,
                                   action="kubectl describe pod x -n ns")])

    class _FakeGraph:
        def resolve(self, ev, ns, rc):
            return invest_only
        def add_provisional(self, *a, **k):
            self.added = True

    fake = _FakeGraph()
    monkeypatch.setattr("src.remediation.remediation_graph.get_graph", lambda: fake)
    monkeypatch.setenv("ESCALATION_BACKEND", "ollama")
    monkeypatch.setattr(escalation, "escalate",
                        lambda rc, ev, ns: [Step(order=0, action_type=COMMAND,
                                                 action="kubectl set image deployment/x c=r/i:2 -n ns")])
    plan = escalation.resolve_with_escalation("ev", "ns", "imagen mala")
    assert plan.source == "escalated"
    assert any(s.action_type == COMMAND for s in plan.steps)
    assert getattr(fake, "added", False) is True


@pytest.mark.unit
def test_resolve_keeps_investigate_plan_when_model_finds_no_command(monkeypatch):
    """Caso externo: el modelo no halla acción segura → se devuelve el plan de
    investigación del catálogo (la nota externa la pone la capa determinista)."""
    from src.remediation.remediation_graph import Plan, Step

    invest_only = Plan(intent="image_auth", namespace="ns",
                       steps=[Step(order=0, action_type=INVESTIGATE,
                                   action="kubectl get secret -n ns")])

    class _FakeGraph:
        def resolve(self, ev, ns, rc):
            return invest_only

    monkeypatch.setattr("src.remediation.remediation_graph.get_graph", lambda: _FakeGraph())
    monkeypatch.setenv("ESCALATION_BACKEND", "ollama")
    monkeypatch.setattr(escalation, "escalate", lambda rc, ev, ns: [])  # sin acción segura
    plan = escalation.resolve_with_escalation("ev", "ns", "auth registry")
    assert plan is invest_only  # se conserva el plan de investigación, sin inventar acción
