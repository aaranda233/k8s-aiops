"""
Tests del clasificador de riesgo kubectl.

CRÍTICO: ningún comando destructivo debe colarse como Level 0 o 1.
Un fallo aquí permitiría ejecución automática de comandos peligrosos.
"""

import pytest

from src.remediation.risk_scorer import score


@pytest.mark.unit
@pytest.mark.parametrize("cmd", [
    "kubectl describe pod nginx -n default",
    "kubectl get pods -n prod",
    "kubectl logs nginx -n default --previous",
    "kubectl top pod nginx -n default",
    "kubectl get events -n prod --sort-by='.lastTimestamp'",
    "kubectl explain deployment",
])
def test_level_0_read_only(cmd):
    assert score(cmd).level == 0


@pytest.mark.unit
@pytest.mark.parametrize("cmd", [
    "kubectl rollout restart deployment/api -n prod",
    "kubectl rollout undo deployment/api -n prod",
    "kubectl scale deployment/api --replicas=3 -n prod",
])
def test_level_1_reversible(cmd):
    assert score(cmd).level == 1


@pytest.mark.unit
@pytest.mark.parametrize("cmd", [
    "kubectl set resources deployment/api --limits=memory=512Mi",
    "kubectl set image deployment/api api=nginx:1.2 -n prod",
    "kubectl set env deployment/api KEY=value",
    "kubectl patch deployment api --patch '{}'",
    "kubectl annotate pod nginx key=value",
    "kubectl label pod nginx env=prod",
])
def test_level_2_config(cmd):
    assert score(cmd).level == 2


@pytest.mark.unit
@pytest.mark.parametrize("cmd", [
    "kubectl delete pod nginx -n prod",
    "kubectl delete deployment api -n prod",
    "kubectl drain node-1",
    "kubectl cordon node-1",
    "kubectl taint nodes node-1 key=value:NoSchedule",
    "kubectl exec nginx -- bash",
    "kubectl apply -f deploy.yaml",
    "kubectl create deployment x --image=nginx",
    "kubectl replace -f deploy.yaml",
    "kubectl run nginx --image=nginx",
])
def test_level_3_destructive_never_auto(cmd):
    """El test más importante: destructivos SIEMPRE Level 3."""
    assert score(cmd).level == 3


@pytest.mark.unit
def test_non_kubectl_is_level_3():
    assert score("rm -rf /").level == 3
    assert score("bash -c 'evil'").level == 3


@pytest.mark.unit
def test_unparseable_is_level_3():
    assert score("kubectl delete 'unclosed quote").level == 3


@pytest.mark.unit
def test_unknown_verb_defaults_conservative():
    """Verbo desconocido → Level 2 (conservador), nunca auto-ejecutado sin aprobación."""
    result = score("kubectl frobnicate deployment/x")
    assert result.level == 2


@pytest.mark.unit
def test_empty_command():
    assert score("kubectl").level == 0


@pytest.mark.unit
def test_delete_cannot_be_misclassified_as_lower():
    """Defensa explícita: verificar que delete NUNCA es 0 ni 1."""
    for variant in [
        "kubectl delete pod x",
        "kubectl   delete   pod   x",
        "kubectl delete -n prod pod x",
    ]:
        assert score(variant).level == 3, f"PELIGRO: {variant!r} no es Level 3"
