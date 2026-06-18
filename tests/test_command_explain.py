"""
Tests del explicador determinista de comandos kubectl (explain_command).

Cada comando del incidente lleva una explicación en español de qué hace y qué
mirar. Determinista (parsea el comando), sin modelo.
"""

import pytest

from src.diagnostics.command_builder import explain_command


@pytest.mark.unit
def test_describe_pod_with_name():
    exp = explain_command("kubectl describe pod postgresql-0 -n postgresql")
    assert "pod" in exp.lower()
    assert "postgresql-0" in exp
    assert "postgresql" in exp  # namespace mencionado


@pytest.mark.unit
def test_describe_pvc_explains_binding():
    exp = explain_command("kubectl describe pvc pvc-data -n ml-abonos-api")
    assert "pvc-data" in exp
    assert "volumen" in exp.lower() or "vincula" in exp.lower()


@pytest.mark.unit
def test_describe_node_no_namespace():
    exp = explain_command("kubectl describe node node-1")
    assert "node-1" in exp
    assert "nodo" in exp.lower()


@pytest.mark.unit
def test_get_secret_mentions_credentials():
    exp = explain_command("kubectl get secret -n postgresql")
    assert "secret" in exp.lower()
    assert "credencial" in exp.lower() or "clave" in exp.lower()


@pytest.mark.unit
def test_get_networkpolicy_mentions_traffic():
    exp = explain_command("kubectl get networkpolicy -n default")
    assert "networkpolic" in exp.lower()
    assert "tráfico" in exp.lower() or "trafico" in exp.lower() or "red" in exp.lower()


@pytest.mark.unit
def test_logs_previous_mentions_crash():
    exp = explain_command("kubectl logs web-abc -n default --previous")
    assert "log" in exp.lower()
    assert "anterior" in exp.lower()


@pytest.mark.unit
def test_get_endpoints():
    exp = explain_command("kubectl get endpoints api -n default")
    assert "endpoint" in exp.lower()


@pytest.mark.unit
def test_rollout_restart_is_reversible():
    exp = explain_command("kubectl rollout restart deployment/web -n default")
    assert "deployment/web" in exp
    assert "reversible" in exp.lower()
    assert "reinicia" in exp.lower()


@pytest.mark.unit
def test_get_events_fallback():
    exp = explain_command("kubectl get events --all-namespaces --sort-by='.lastTimestamp'")
    assert "evento" in exp.lower()


@pytest.mark.unit
def test_non_kubectl_returns_empty():
    assert explain_command("") == ""
    assert explain_command("echo hola") == ""


@pytest.mark.unit
def test_generic_verb_fallback():
    exp = explain_command("kubectl get configmap -n default")
    assert exp  # no vacío
    assert exp.startswith(("Lista", "Muestra"))
