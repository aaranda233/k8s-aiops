"""
Tests de la clasificación App / Plataforma de incidentes (classify_category).

App       → código/config/credenciales/salud de la aplicación.
Plataforma → nodo/recursos/almacenamiento/red/imagen (infra).
Determinista, según la intención detectada en la evidencia.
"""

import pytest

from src.diagnostics.command_builder import classify_category


@pytest.mark.unit
@pytest.mark.parametrize("evidence,root_cause", [
    ("intranet Pod/x OOMKilling Memory cgroup out of memory", "OOMKilled"),
    ("ns Pod/x Evicted The node was low on resource: memory. Node node-1", "memory pressure nodo"),
    ("ns Pod/x FailedBinding PVC pvc-data pending: no volume", "PVC sin volumen"),
    ("default Pod/api connection refused by NetworkPolicy denied", "NetworkPolicy bloquea"),
    ("argocd Pod/x ErrImagePull manifest not in registry, back-off pulling image", "imagen no encontrada"),
    ("ns Pod/x FailedScheduling insufficient cpu, unschedulable", "cpu insuficiente"),
])
def test_platform_incidents(evidence, root_cause):
    assert classify_category(evidence, root_cause) == "platform"


@pytest.mark.unit
@pytest.mark.parametrize("evidence,root_cause", [
    ("postgresql Pod/postgresql-0 FATAL: role does not exist", "rol de postgresql no existe"),
    ("default Pod/web CrashLoopBackOff: missing env var in configmap, exit code 1", "config mal"),
    ("default Pod/api Unhealthy Liveness probe failed", "probe de liveness falla"),
    ("default Pod/api Service has no endpoints, selector mismatch", "service sin endpoints"),
])
def test_app_incidents(evidence, root_cause):
    assert classify_category(evidence, root_cause) == "app"


@pytest.mark.unit
def test_unknown_defaults_to_app():
    assert classify_category("default Pod/x algo raro sin patrón", "causa desconocida") == "app"
