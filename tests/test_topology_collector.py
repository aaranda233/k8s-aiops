"""
Tests del constructor de topología.

Cliente k8s mockeado. Verifica las relaciones del grafo
(ingress→service→pod→node), la detección de salud y el cacheo.
"""

from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest

from src.collector.topology_collector import (
    HEALTH_ERROR,
    HEALTH_OK,
    HEALTH_WARN,
    TopologyCollector,
    _pod_health,
)


def _collector():
    with patch("src.collector.topology_collector.k8s_config"), \
         patch("src.collector.topology_collector.client"):
        c = TopologyCollector(cache_ttl=0)
    c._v1 = MagicMock()
    c._net = MagicMock()
    return c


def _meta(name, ns=None):
    return NS(name=name, namespace=ns)


@pytest.mark.unit
def test_graph_relates_ingress_service_pod_node():
    c = _collector()
    # 1 nodo k8s
    c._v1.list_node.return_value = NS(items=[
        NS(metadata=_meta("worker-1"), status=NS(conditions=[NS(type="Ready", status="True")])),
    ])
    # 1 pod en worker-1, Running ready
    pod = NS(
        metadata=_meta("api-123", "prod"),
        spec=NS(node_name="worker-1"),
        status=NS(phase="Running", container_statuses=[NS(ready=True, state=NS(waiting=None), restart_count=0)]),
    )
    c._v1.list_pod_for_all_namespaces.return_value = NS(items=[pod])
    # 1 service
    c._v1.list_service_for_all_namespaces.return_value = NS(items=[
        NS(metadata=_meta("api", "prod"), spec=NS(type="ClusterIP")),
    ])
    # endpoints: api → pod api-123
    c._v1.list_endpoints_for_all_namespaces.return_value = NS(items=[
        NS(metadata=_meta("api", "prod"),
           subsets=[NS(addresses=[NS(target_ref=NS(kind="Pod", name="api-123"))])]),
    ])
    # ingress → service api
    c._net.list_ingress_for_all_namespaces.return_value = NS(items=[
        NS(metadata=_meta("web", "prod"), spec=NS(rules=[
            NS(host="app.example.com", http=NS(paths=[
                NS(backend=NS(service=NS(name="api"))),
            ])),
        ])),
    ])

    g = c.build_graph()
    ids = {n["id"] for n in g["nodes"]}
    assert "node/worker-1" in ids
    assert "pod/prod/api-123" in ids
    assert "svc/prod/api" in ids
    assert "ing/prod/web" in ids

    kinds = {(l["source"], l["target"], l["kind"]) for l in g["links"]}
    assert ("pod/prod/api-123", "node/worker-1", "runs-on") in kinds
    assert ("svc/prod/api", "pod/prod/api-123", "serves") in kinds
    assert ("ing/prod/web", "svc/prod/api", "routes") in kinds
    assert g["stats"]["pods"] == 1 and g["stats"]["services"] == 1


@pytest.mark.unit
def test_service_without_endpoints_is_warn():
    c = _collector()
    c._v1.list_node.return_value = NS(items=[])
    c._v1.list_pod_for_all_namespaces.return_value = NS(items=[])
    c._v1.list_service_for_all_namespaces.return_value = NS(items=[
        NS(metadata=_meta("orphan", "prod"), spec=NS(type="ClusterIP")),
    ])
    c._v1.list_endpoints_for_all_namespaces.return_value = NS(items=[
        NS(metadata=_meta("orphan", "prod"), subsets=[]),
    ])
    c._net.list_ingress_for_all_namespaces.return_value = NS(items=[])
    g = c.build_graph()
    svc = next(n for n in g["nodes"] if n["id"] == "svc/prod/orphan")
    assert svc["health"] == HEALTH_WARN


@pytest.mark.unit
def test_pod_health_detection():
    # CrashLoopBackOff → error
    crash = NS(status=NS(phase="Running", container_statuses=[
        NS(ready=False, restart_count=10, state=NS(waiting=NS(reason="CrashLoopBackOff"))),
    ]))
    assert _pod_health(crash) == HEALTH_ERROR
    # Failed → error
    failed = NS(status=NS(phase="Failed", container_statuses=[]))
    assert _pod_health(failed) == HEALTH_ERROR
    # Running ready → ok
    ok = NS(status=NS(phase="Running", container_statuses=[
        NS(ready=True, restart_count=0, state=NS(waiting=None)),
    ]))
    assert _pod_health(ok) == HEALTH_OK
    # Pending → warn
    pending = NS(status=NS(phase="Pending", container_statuses=[]))
    assert _pod_health(pending) == HEALTH_WARN


@pytest.mark.unit
def test_cache_avoids_rebuild():
    c = _collector()
    c._cache_ttl = 999
    for m in (c._v1.list_node, c._v1.list_pod_for_all_namespaces,
              c._v1.list_service_for_all_namespaces, c._v1.list_endpoints_for_all_namespaces):
        m.return_value = NS(items=[])
    c._net.list_ingress_for_all_namespaces.return_value = NS(items=[])
    c.build_graph()
    c.build_graph()  # segunda vez → cache
    assert c._v1.list_node.call_count == 1  # solo se construyó una vez


@pytest.mark.unit
def test_partial_api_failure_does_not_crash():
    c = _collector()
    c._v1.list_node.side_effect = Exception("nodes API down")
    c._v1.list_pod_for_all_namespaces.return_value = NS(items=[])
    c._v1.list_service_for_all_namespaces.return_value = NS(items=[])
    c._v1.list_endpoints_for_all_namespaces.return_value = NS(items=[])
    c._net.list_ingress_for_all_namespaces.return_value = NS(items=[])
    g = c.build_graph()  # no debe lanzar
    assert "nodes" in g and "links" in g
