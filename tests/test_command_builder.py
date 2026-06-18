"""
Tests del constructor de comandos kubectl (src/diagnostics/command_builder.py).

Verifica: extracción de recursos desde la evidencia, catálogo intención→comando
(verbo alineado con el escenario), override del namespace al culpable, y rechazo
de comandos frágiles del modelo. El objetivo es subir kubectl_ns_ok / verb_ok.
"""

import pytest

from src.diagnostics.command_builder import (
    build_command,
    extract_node,
    extract_pod,
    extract_pvc,
)

# ── Extracción de recursos ──────────────────────────────────────────────────

@pytest.mark.unit
def test_extract_pod_from_log_line():
    ev = "postgresql Pod/postgresql-7f7d545cb8-dhdrf FATAL: role does not exist"
    assert extract_pod(ev) == "postgresql-7f7d545cb8-dhdrf"


@pytest.mark.unit
def test_extract_node_from_event():
    ev = "intranet Pod/scheduler-x Evicted ... Node node-1 status is now: NodeHasSufficientMemory"
    assert extract_node(ev) == "node-1"


@pytest.mark.unit
def test_extract_pvc():
    ev = "ml-abonos-api Pod/x FailedBinding PVC pvc-report-generator-data is pending"
    assert extract_pvc(ev) == "pvc-report-generator-data"


# ── Catálogo intención→comando (verbo + recurso + namespace) ────────────────

@pytest.mark.unit
def test_oom_builds_describe_pod_in_namespace():
    ev = "intranet Pod/scheduler-j7mwc OOMKilling Memory cgroup out of memory: Kill process"
    cmd = build_command(ev, namespace="intranet", root_cause="OOMKilled en el pod")
    assert cmd.startswith("kubectl describe ")
    assert "-n intranet" in cmd
    assert "scheduler-j7mwc" in cmd


@pytest.mark.unit
def test_pvc_builds_describe_pvc():
    ev = "ml-abonos-api Pod/x FailedBinding PVC pvc-report-generator-data pending: no volume"
    cmd = build_command(ev, namespace="ml-abonos-api", root_cause="PVC sin volumen")
    assert "describe pvc" in cmd
    assert "pvc-report-generator-data" in cmd
    assert "-n ml-abonos-api" in cmd


@pytest.mark.unit
def test_network_policy_builds_get_networkpolicy():
    ev = "default Pod/api connection refused by NetworkPolicy, traffic denied"
    cmd = build_command(ev, namespace="default", root_cause="NetworkPolicy bloquea")
    assert "get networkpolicy" in cmd
    assert "-n default" in cmd


@pytest.mark.unit
def test_node_pressure_builds_describe_node_without_namespace():
    ev = "intranet Pod/x Evicted The node was low on resource: memory. Node node-1 status"
    cmd = build_command(ev, namespace="intranet", root_cause="memory pressure en el nodo")
    assert "describe node" in cmd
    assert "node-1" in cmd
    assert " -n " not in cmd  # los nodos NO son namespaced


@pytest.mark.unit
def test_crash_config_builds_logs_previous():
    ev = "default Pod/web-abc CrashLoopBackOff: missing env var in configmap, exit code 1"
    cmd = build_command(ev, namespace="default", root_cause="CrashLoop por configmap")
    assert cmd.startswith("kubectl logs ")
    assert "-n default" in cmd
    assert "--previous" in cmd


# ── Override de namespace + rechazo de comandos frágiles ────────────────────

@pytest.mark.unit
def test_model_command_namespace_is_corrected():
    """Si el modelo apunta al namespace equivocado, se corrige al culpable."""
    ev = "postgresql Pod/postgresql-0 OOMKilling out of memory"
    bad = "kubectl describe pod postgresql-0 -n aiops-demo"
    cmd = build_command(ev, namespace="postgresql", root_cause="OOM", model_cmd=bad)
    assert "-n postgresql" in cmd
    assert "-n aiops-demo" not in cmd


@pytest.mark.unit
def test_fragile_model_command_is_replaced():
    """Comandos con substitución $(...) o pipes se descartan por el determinista."""
    ev = "postgresql Pod/postgresql-0 OOMKilling out of memory"
    fragile = "kubectl logs -n postgresql $(kubectl get pod -l app=postgresql -o name)"
    cmd = build_command(ev, namespace="postgresql", root_cause="OOM", model_cmd=fragile)
    assert "$(" not in cmd
    assert cmd.startswith("kubectl ")


@pytest.mark.unit
def test_command_never_has_placeholders():
    ev = "default Pod/x something weird happened"
    cmd = build_command(ev, namespace="default", root_cause="causa desconocida")
    assert "<" not in cmd and ">" not in cmd
    assert cmd.startswith("kubectl ")
