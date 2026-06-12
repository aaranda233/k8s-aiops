"""
Tests del toolbox kubectl de solo lectura (usado en la fase de investigación).

CRÍTICO: ningún verbo de escritura debe poder ejecutarse a través del toolbox.
"""

import pytest

from src.diagnostics.kubectl_toolbox import execute


@pytest.mark.unit
@pytest.mark.parametrize("cmd", [
    "kubectl delete pod nginx",
    "kubectl apply -f deploy.yaml",
    "kubectl patch deployment x --patch '{}'",
    "kubectl scale deployment x --replicas=0",
    "kubectl exec nginx -- bash",
    "kubectl drain node-1",
    "kubectl edit deployment x",
])
def test_write_verbs_rejected(cmd):
    result = execute(cmd)
    assert result.returncode == 1
    assert result.error is not None
    assert "prohibido" in result.error.lower() or "permite" in result.error.lower()


@pytest.mark.unit
def test_non_kubectl_rejected():
    result = execute("rm -rf /")
    assert result.returncode == 1
    assert "kubectl" in result.error.lower()


@pytest.mark.unit
def test_forbidden_flag_rejected():
    result = execute("kubectl get pods -f deploy.yaml")
    assert result.returncode == 1


@pytest.mark.unit
def test_unparseable_rejected():
    result = execute("kubectl get 'unclosed")
    assert result.returncode == 1
