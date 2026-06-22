"""Tests del modo B: aprobación → ejecución → verificación por re-detección.

Verifica las transiciones de estado del incident_store (executed/resolved/failed)
y la barrera de seguridad del executor (solo L0/L1). No ejecuta kubectl real.
"""

import time

import pytest

from src.remediation.executor import execute_if_reversible
from src.remediation.incident_store import (
    STATUS_APPROVED,
    STATUS_EXECUTED,
    STATUS_FAILED,
    STATUS_RESOLVED,
    Incident,
    IncidentStore,
)


def _store_with(status: str):
    s = IncidentStore()
    inc = Incident(
        id="INC-1", created_at=time.time(), namespaces=["web"], score=1.0,
        root_cause="x", kubectl_cmd="kubectl get pods -n web", risk_level=1,
        risk_label="reversible", status=status, solution_source="graph",
        solution_key="network",
    )
    s._incidents[inc.id] = inc
    return s, inc


@pytest.mark.unit
def test_mark_executed_success_sets_executed():
    s, _ = _store_with(STATUS_APPROVED)
    s.mark_executed("INC-1", "rollout restarted", success=True)
    inc = s.get("INC-1")
    assert inc.status == STATUS_EXECUTED
    assert inc.execution_output == "rollout restarted"
    assert inc.verified is None  # aún sin verificar


@pytest.mark.unit
def test_mark_executed_failure_sets_failed():
    s, _ = _store_with(STATUS_APPROVED)
    s.mark_executed("INC-1", "error", success=False)
    inc = s.get("INC-1")
    assert inc.status == STATUS_FAILED
    assert inc.verified is False


@pytest.mark.unit
def test_recurrence_marks_executed_as_failed():
    s, inc = _store_with(STATUS_EXECUTED)
    s.fail_executed("INC-1")
    assert s.get("INC-1").status == STATUS_FAILED
    assert s.get("INC-1").verified is False


@pytest.mark.unit
def test_sweep_resolves_after_grace():
    s, inc = _store_with(STATUS_EXECUTED)
    inc.last_seen = time.time() - 1000  # ejecutado hace rato, sin recurrencia
    s.sweep_resolved(grace_seconds=300)
    assert s.get("INC-1").status == STATUS_RESOLVED
    assert s.get("INC-1").verified is True


@pytest.mark.unit
def test_sweep_does_not_resolve_within_grace():
    s, inc = _store_with(STATUS_EXECUTED)
    inc.last_seen = time.time()
    s.sweep_resolved(grace_seconds=300)
    assert s.get("INC-1").status == STATUS_EXECUTED  # aún en periodo de gracia


@pytest.mark.unit
def test_executor_gate_blocks_destructive():
    assert execute_if_reversible("kubectl delete pod web-0 -n web") is None
    assert execute_if_reversible("kubectl apply -f x.yaml") is None
    # L0/L1 sí intentan ejecutar (devuelven un ExecutionResult, no None)
    assert execute_if_reversible("kubectl get pods -n web") is not None
    assert execute_if_reversible("kubectl rollout restart deployment/web -n web") is not None


@pytest.mark.unit
def test_sweep_fires_graph_verification_hook():
    s, inc = _store_with(STATUS_EXECUTED)
    inc.last_seen = time.time() - 1000
    captured = {}
    s.set_feedback_hook(lambda d: captured.update(d))
    s.sweep_resolved(grace_seconds=300)
    assert captured.get("status") == STATUS_RESOLVED
    assert captured.get("verified") is True
    assert captured.get("solution_key") == "network"  # el hook puede verificar el grafo
