"""
Tests del colector de logs de aplicación.

Cliente k8s mockeado. Verifica el mapeo a LogEntry, la detección de nivel,
y CRÍTICO: las salvaguardas de seguridad (namespaces obligatorios, solo lectura,
acotado por tail/max_pods).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.collector.k8s_collector import LogEntry
from src.collector.log_collector import LogCollector, _detect_level


@pytest.mark.unit
@pytest.mark.parametrize("line,expected", [
    ("2026-01-01 ERROR connection refused", "ERROR"),
    ("FATAL out of memory", "FATAL"),
    ("WARN deprecated api", "WARN"),
    ("INFO request handled", "INFO"),
    ("plain message no level", "LOG"),
])
def test_detect_level(line, expected):
    assert _detect_level(line) == expected


@pytest.mark.unit
def test_empty_namespaces_means_all_cluster():
    """Vacío = todo el cluster (list_pod_for_all_namespaces)."""
    with patch("src.collector.log_collector.k8s_config"), \
         patch("src.collector.log_collector.client"):
        c = LogCollector(namespaces=[])
        c._v1 = MagicMock()
        assert c.all_namespaces is True
        pods = MagicMock()
        p = MagicMock()
        p.metadata.namespace = "kube-system"
        p.metadata.name = "coredns-1"
        cont = MagicMock()
        cont.name = "coredns"
        p.spec.containers = [cont]
        pods.items = [p]
        c._v1.list_pod_for_all_namespaces.return_value = pods
        targets = c._list_target_pods()
        assert targets == [("kube-system", "coredns-1", ["coredns"])]
        c._v1.list_pod_for_all_namespaces.assert_called_once()


@pytest.mark.unit
def test_multi_container_pod_reads_each_container():
    """Pods multi-contenedor: se leen TODOS los contenedores (antes se saltaban)."""
    with patch("src.collector.log_collector.k8s_config"), \
         patch("src.collector.log_collector.client"):
        c = LogCollector(namespaces=[])
        c._v1 = MagicMock()
        p = MagicMock()
        p.metadata.namespace = "longhorn-system"
        p.metadata.name = "csi-plugin"
        c1, c2 = MagicMock(), MagicMock()
        c1.name, c2.name = "plugin", "registrar"
        p.spec.containers = [c1, c2]
        pods = MagicMock()
        pods.items = [p]
        c._v1.list_pod_for_all_namespaces.return_value = pods
        c._v1.read_namespaced_pod_log.return_value = _resp("ERROR boom\n")
        list(c._poll_once())
        # Se leyó una vez por cada contenedor
        containers_read = {call.kwargs.get("container") for call in c._v1.read_namespaced_pod_log.call_args_list}
        assert containers_read == {"plugin", "registrar"}


@pytest.mark.unit
def test_explicit_namespaces_scopes_to_them():
    with patch("src.collector.log_collector.k8s_config"), \
         patch("src.collector.log_collector.client"):
        c = LogCollector(namespaces=["prod"])
        assert c.all_namespaces is False


def _collector():
    with patch("src.collector.log_collector.k8s_config"), \
         patch("src.collector.log_collector.client"):
        c = LogCollector(namespaces=["prod"], tail_lines=10, max_pods=5)
    c._v1 = MagicMock()
    return c


def _resp(text: str):
    """Simula la respuesta con _preload_content=False (tiene .data en bytes)."""
    r = MagicMock()
    r.data = text.encode("utf-8")
    return r


@pytest.mark.unit
def test_read_pod_logs_maps_to_logentry():
    c = _collector()
    c._v1.read_namespaced_pod_log.return_value = _resp("ERROR boom\nINFO ok\n\n")
    entries = list(c._read_pod_logs("prod", "api-123"))
    assert len(entries) == 2  # la línea vacía se descarta
    assert all(isinstance(e, LogEntry) for e in entries)
    assert entries[0].reason == "ERROR"
    assert entries[0].namespace == "prod"
    assert entries[0].source == "Pod/api-123"
    assert entries[0].event_type == "LOG"
    assert "ERROR boom" in entries[0].message


@pytest.mark.unit
def test_read_pod_logs_uses_readonly_bounded_call():
    """Verifica read (solo lectura) con since_seconds, tail_lines y _preload_content=False."""
    c = _collector()
    c._v1.read_namespaced_pod_log.return_value = _resp("linea")
    list(c._read_pod_logs("prod", "api-123"))
    kwargs = c._v1.read_namespaced_pod_log.call_args.kwargs
    assert kwargs["namespace"] == "prod"
    assert kwargs["tail_lines"] == 10
    assert kwargs["since_seconds"] == c.since_seconds
    assert kwargs["_preload_content"] is False


@pytest.mark.unit
def test_pod_log_error_is_skipped_not_raised():
    from kubernetes.client.rest import ApiException
    c = _collector()
    c._v1.read_namespaced_pod_log.side_effect = ApiException(status=400)
    entries = list(c._read_pod_logs("prod", "terminating-pod"))
    assert entries == []  # no lanza, solo salta


@pytest.mark.unit
def test_poll_respects_max_pods():
    c = _collector()  # max_pods=5
    # 10 pods listados, pero solo se deben leer 5
    pods = MagicMock()
    pods.items = [MagicMock(metadata=MagicMock(name=f"p{i}")) for i in range(10)]
    for i, p in enumerate(pods.items):
        p.metadata.name = f"pod-{i}"
    c._v1.list_namespaced_pod.return_value = pods
    c._v1.read_namespaced_pod_log.return_value = _resp("log")
    list(c._poll_once())
    assert c._v1.read_namespaced_pod_log.call_count <= 5
