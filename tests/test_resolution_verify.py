"""Verificación de resolución por re-detección (Modo B / sweep_resolved).

Confirma que un incidente solo se da por RESUELTO si la anomalía **no recurre**
(señal de outcome real), no por que el comando se ejecutara. Es la comprobación de
"¿se ha solucionado de verdad?" tras aplicar la remediación (botón play → executed →
el pipeline barre y resuelve si no reaparece).
"""

import time

import pytest

from src.remediation.incident_store import (
    STATUS_EXECUTED,
    STATUS_PENDING,
    STATUS_RESOLVED,
    Incident,
    IncidentStore,
)


def _inc(iid: str, status: str) -> Incident:
    return Incident(
        id=iid, created_at=time.time(), namespaces=["argocd"], score=0.8,
        root_cause="proxy responde mal", kubectl_cmd="kubectl get pods -n argocd",
        risk_level=1, risk_label="reversible", status=status,
    )


@pytest.mark.unit
def test_sweep_resolves_when_anomaly_does_not_recur():
    """EXECUTED + la anomalía no reaparece en 'grace' → RESOLVED + verificado."""
    store = IncidentStore()
    store.add(_inc("INC-A", STATUS_EXECUTED))
    store.get("INC-A").last_seen = time.time() - 120  # no re-visto hace 120 s
    store.sweep_resolved(grace_seconds=90)
    assert store.get("INC-A").status == STATUS_RESOLVED
    assert store.get("INC-A").verified is True


@pytest.mark.unit
def test_sweep_keeps_executed_when_anomaly_recurs():
    """EXECUTED + la anomalía vuelve a detectarse (bump) → sigue EXECUTED (no resuelto)."""
    store = IncidentStore()
    store.add(_inc("INC-B", STATUS_EXECUTED))
    store.bump("INC-B")  # re-detección reciente → last_seen ahora
    store.sweep_resolved(grace_seconds=90)
    assert store.get("INC-B").status == STATUS_EXECUTED
    assert store.get("INC-B").verified is None


@pytest.mark.unit
def test_sweep_ignores_non_executed_incidents():
    store = IncidentStore()
    store.add(_inc("INC-C", STATUS_PENDING))
    store.get("INC-C").last_seen = time.time() - 999
    store.sweep_resolved(grace_seconds=90)
    assert store.get("INC-C").status == STATUS_PENDING
