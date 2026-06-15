"""Tests del escáner de seguridad. Cliente k8s mockeado."""

from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest

from src.security.scanner import (
    SEV_CRITICAL,
    SEV_HIGH,
    SEV_MEDIUM,
    SecurityScanner,
)


def _scanner():
    with patch("src.security.scanner.k8s_config"), patch("src.security.scanner.client"):
        s = SecurityScanner(cache_ttl=0)
    s._v1 = MagicMock(); s._rbac = MagicMock(); s._net = MagicMock()
    s._rbac.list_cluster_role_binding.return_value = NS(items=[])
    s._net.list_network_policy_for_all_namespaces.return_value = NS(items=[])
    return s


def _container(name="app", image="nginx:1.25", privileged=False, run_as_user=None,
               caps=None, env=None, limits={"memory": "256Mi"}):
    sc = NS(privileged=privileged, run_as_user=run_as_user,
            capabilities=NS(add=caps) if caps else None)
    return NS(name=name, image=image, security_context=sc,
              env=env or [], resources=NS(limits=limits))


def _pod(ns="prod", name="api-1", containers=None, host_network=False, host_pid=False,
         host_ipc=False, volumes=None, pod_sc=None):
    return NS(metadata=NS(namespace=ns, name=name),
              spec=NS(containers=containers or [_container()], init_containers=[],
                      host_network=host_network, host_pid=host_pid, host_ipc=host_ipc,
                      volumes=volumes or [], security_context=pod_sc))


def _run(scanner, pods):
    scanner._v1.list_pod_for_all_namespaces.return_value = NS(items=pods)
    return scanner.scan()["findings"]


def _titles(findings):
    return [f["title"] for f in findings]


@pytest.mark.unit
def test_privileged_is_critical():
    s = _scanner()
    f = _run(s, [_pod(containers=[_container(privileged=True)])])
    crit = [x for x in f if x["severity"] == SEV_CRITICAL]
    assert any("privilegiado" in x["title"] for x in crit)


@pytest.mark.unit
def test_run_as_root_is_high():
    s = _scanner()
    f = _run(s, [_pod(containers=[_container(run_as_user=0)])])
    assert any(x["severity"] == SEV_HIGH and "root" in x["title"] for x in f)


@pytest.mark.unit
def test_host_network_flagged():
    s = _scanner()
    f = _run(s, [_pod(host_network=True)])
    assert any("hostNetwork" in t for t in _titles(f))


@pytest.mark.unit
def test_hostpath_volume_flagged():
    s = _scanner()
    vol = NS(host_path=NS(path="/var/run/docker.sock"))
    f = _run(s, [_pod(volumes=[vol])])
    assert any("hostPath" in t for t in _titles(f))


@pytest.mark.unit
def test_dangerous_capability_flagged():
    s = _scanner()
    f = _run(s, [_pod(containers=[_container(caps=["SYS_ADMIN"])])])
    assert any("SYS_ADMIN" in t for t in _titles(f))


@pytest.mark.unit
def test_latest_image_flagged():
    s = _scanner()
    f = _run(s, [_pod(containers=[_container(image="myapp:latest")])])
    assert any("tag mutable" in t for t in _titles(f))
    # imagen con tag fijo no se marca
    s2 = _scanner()
    f2 = _run(s2, [_pod(containers=[_container(image="myapp:v1.2.3")])])
    assert not any("tag mutable" in t for t in _titles(f2))


@pytest.mark.unit
def test_hardcoded_secret_in_env_flagged():
    s = _scanner()
    env = [NS(name="DB_PASSWORD", value="hunter2")]
    f = _run(s, [_pod(containers=[_container(env=env)])])
    assert any("secreto en variable" in t for t in _titles(f))
    # secret via valueFrom (value=None) no se marca
    s2 = _scanner()
    env2 = [NS(name="DB_PASSWORD", value=None)]
    f2 = _run(s2, [_pod(containers=[_container(env=env2)])])
    assert not any("secreto en variable" in t for t in _titles(f2))


@pytest.mark.unit
def test_no_memory_limit_flagged():
    s = _scanner()
    f = _run(s, [_pod(containers=[_container(limits=None)])])
    assert any(x["severity"] == SEV_MEDIUM and "límite de memoria" in x["title"] for x in f)


@pytest.mark.unit
def test_cluster_admin_binding_is_critical():
    s = _scanner()
    s._v1.list_pod_for_all_namespaces.return_value = NS(items=[])
    s._rbac.list_cluster_role_binding.return_value = NS(items=[
        NS(metadata=NS(name="dev-admin"), role_ref=NS(name="cluster-admin"),
           subjects=[NS(kind="ServiceAccount", namespace="prod", name="deployer")]),
    ])
    findings = s.scan()["findings"]
    assert any(x["severity"] == SEV_CRITICAL and "cluster-admin" in x["title"] for x in findings)


@pytest.mark.unit
def test_system_cluster_admin_not_flagged():
    s = _scanner()
    s._v1.list_pod_for_all_namespaces.return_value = NS(items=[])
    s._rbac.list_cluster_role_binding.return_value = NS(items=[
        NS(metadata=NS(name="sys"), role_ref=NS(name="cluster-admin"),
           subjects=[NS(kind="Group", namespace=None, name="system:masters")]),
    ])
    assert not s.scan()["findings"]


@pytest.mark.unit
def test_namespace_without_netpol_flagged():
    s = _scanner()
    s._v1.list_pod_for_all_namespaces.return_value = NS(items=[_pod(ns="prod")])
    findings = s.scan()["findings"]
    assert any("sin NetworkPolicy" in t for t in _titles(findings))


@pytest.mark.unit
def test_summary_counts():
    s = _scanner()
    f = _run(s, [_pod(containers=[_container(privileged=True, run_as_user=0)])])
    summary = s.scan()["summary"]
    assert summary["total"] == len(f)
    assert summary["critical"] >= 1 and summary["high"] >= 1


@pytest.mark.unit
def test_clean_pod_minimal_findings():
    s = _scanner()
    # pod bien configurado: tag fijo, no root, con límites, sin host*
    clean = _container(image="app:v1", run_as_user=1000, limits={"memory": "256Mi", "cpu": "100m"})
    f = _run(s, [_pod(ns="prod", containers=[clean])])
    # solo debería aparecer (a lo sumo) el aviso de NetworkPolicy del namespace
    titles = _titles(f)
    assert all("sin NetworkPolicy" in t for t in titles) or not titles
