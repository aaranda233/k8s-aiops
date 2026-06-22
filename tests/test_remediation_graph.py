"""Tests del grafo de conocimiento de remediación (src/remediation/remediation_graph.py).

Verifica: semilla no vacía, resolución a plan multi-paso (caso ingress y secret),
binding del namespace/workload, y que un miss devuelve None.
"""

import pytest

from src.remediation.remediation_graph import (
    COMMAND,
    GUIDANCE,
    INVESTIGATE,
    RemediationGraph,
    Step,
    verify_from_incident,
)


@pytest.fixture
def graph(tmp_path):
    g = RemediationGraph(db_path=str(tmp_path / "g.db"))
    g.seed_from_catalog()
    return g


@pytest.mark.unit
def test_seed_not_empty(graph):
    s = graph.stats()
    assert s["nodes"] >= 10
    assert s["edges"] >= 20


@pytest.mark.unit
def test_seed_is_idempotent(tmp_path):
    g = RemediationGraph(db_path=str(tmp_path / "g.db"))
    g.seed_from_catalog()
    n1 = g.stats()
    g.seed_from_catalog()
    assert g.stats() == n1


@pytest.mark.unit
def test_network_ingress_is_multistep(graph):
    ev = ("haproxy-ingress Pod/haproxy-ingress-controller-abc123de-x4k2p "
          "connection refused, traffic denied")
    plan = graph.resolve(ev, namespace="haproxy-ingress", root_cause="fallo de red en el ingress")
    assert plan is not None
    assert plan.source == "graph"
    # multi-paso: NO es un solo comando
    assert len(plan.steps) >= 3
    types = [s.action_type for s in plan.steps]
    assert INVESTIGATE in types
    # primero investiga el backend (endpoints), no reinicia a ciegas
    assert "endpoints" in plan.steps[0].action
    # el último recurso es el rollout restart, enlazado al workload + namespace
    restart = [s for s in plan.steps if s.action_type == COMMAND]
    assert restart and restart[0].action == (
        "kubectl rollout restart deployment/haproxy-ingress-controller -n haproxy-ingress"
    )


@pytest.mark.unit
def test_crash_secret_plan_has_guidance_and_namespace(graph):
    ev = "postgresql Pod/postgresql-0 FATAL: role does not exist"
    plan = graph.resolve(ev, namespace="postgresql", root_cause="el rol no existe")
    assert plan is not None
    assert plan.intent == "crash_secret"
    # contiene una guía manual (crear/corregir el secret)
    assert any(s.action_type == GUIDANCE for s in plan.steps)
    # los comandos llevan el namespace correcto
    for s in plan.steps:
        if s.action_type != GUIDANCE:
            assert "-n postgresql" in s.action or s.action.startswith("kubectl describe node")


@pytest.mark.unit
def test_step_without_resource_is_dropped(graph):
    # crash_config sin Pod/ en la evidencia → el paso `logs {pod}` y el restart
    # {workload} se descartan; queda al menos la guía.
    ev = "default crashloopbackoff exit code 1 in configmap"
    plan = graph.resolve(ev, namespace="default", root_cause="crashloop por configmap")
    assert plan is not None
    assert all("{" not in s.action for s in plan.steps)  # sin placeholders sin resolver


@pytest.mark.unit
def test_root_cause_wins_over_noisy_evidence(graph):
    """Evidencia ruidosa de ventana multi-namespace con 'connection refused'
    NO debe arrastrar a `network` si el root_cause es claro (rol/secret)."""
    rc = "El error indica que el rol de usuario no existe en la base de datos PostgreSQL."
    noisy = ("postgresql Pod/postgresql-0 FATAL: role does not exist\n"
             "otra-app connection refused to postgresql\n"
             "default i/o timeout connecting")
    plan = graph.resolve(noisy, namespace="postgresql", root_cause=rc)
    assert plan is not None
    assert plan.intent == "crash_secret"  # no 'network'
    assert any("secret" in s.action for s in plan.steps)


@pytest.mark.unit
def test_miss_returns_none(graph):
    plan = graph.resolve("algo totalmente ajeno xyzqwerty", namespace="x", root_cause="")
    assert plan is None


@pytest.mark.unit
def test_mark_verified_updates_counts(graph):
    graph.mark_verified("crash_secret", success=True)
    node = graph._conn.execute("SELECT id FROM nodes WHERE intent=?", ("crash_secret",)).fetchone()
    rows = graph._conn.execute(
        "SELECT verified, success_count, attempt_count FROM edges WHERE node_id=?",
        (node["id"],),
    ).fetchall()
    assert rows and all(r["verified"] == 1 for r in rows)
    assert all(r["success_count"] == 1 and r["attempt_count"] == 1 for r in rows)


@pytest.mark.unit
def test_verify_from_incident_success(graph):
    graph.add_provisional("escalated:abc", [Step(0, COMMAND, "kubectl get pods -n x")],
                          signature_text="algo nuevo")
    inc = {"solution_source": "escalated", "solution_key": "escalated:abc",
           "status": "resolved", "response": "approved", "verified": None}
    assert verify_from_incident(graph, inc) is True
    node = graph._conn.execute("SELECT id FROM nodes WHERE intent=?", ("escalated:abc",)).fetchone()
    r = graph._conn.execute("SELECT verified FROM edges WHERE node_id=?", (node["id"],)).fetchone()
    assert r["verified"] == 1


@pytest.mark.unit
def test_verify_from_incident_failure_and_noop(graph):
    # fallo → False
    inc_fail = {"solution_source": "graph", "solution_key": "crash_secret",
                "status": "failed", "response": None, "verified": None}
    assert verify_from_incident(graph, inc_fail) is False
    # catálogo (no del grafo) → no aplica
    inc_cat = {"solution_source": "catalog", "solution_key": "", "status": "resolved"}
    assert verify_from_incident(graph, inc_cat) is None


@pytest.mark.unit
def test_command_steps_are_reversible(graph):
    """Ningún paso COMMAND del catálogo debe ser destructivo (risk_level 3)."""
    ev = "ns Pod/app-abc12345de-x4k2p OOMKilled out of memory"
    plan = graph.resolve(ev, namespace="ns", root_cause="oom")
    assert plan is not None
    for s in plan.steps:
        if s.action_type == COMMAND:
            assert s.risk_level <= 1
            assert "rollout restart" in s.action
