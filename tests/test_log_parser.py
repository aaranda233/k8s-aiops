"""
Tests del parser de logs online con Drain3 (src/parser/log_parser.py).

Verifica el templating (abstracción de tokens dinámicos), la agrupación de logs
similares bajo un mismo cluster_id, y el enmascarado de IPs/UUIDs/números.
"""

import pytest

from src.parser.log_parser import LogParser, ParsedLog


@pytest.mark.unit
def test_parse_returns_parsedlog_with_metadata():
    p = LogParser()
    out = p.parse("Started container nginx", namespace="default", timestamp=123.0)
    assert isinstance(out, ParsedLog)
    assert out.raw == "Started container nginx"
    assert out.namespace == "default"
    assert out.timestamp == 123.0
    assert isinstance(out.cluster_id, int)


@pytest.mark.unit
def test_parse_carries_level():
    p = LogParser()
    out = p.parse("FATAL: meltdown", namespace="prod", level="FATAL")
    assert out.level == "FATAL"
    # Sin nivel, queda vacío
    assert p.parse("algo normal").level == ""


@pytest.mark.unit
def test_similar_logs_share_cluster_id():
    p = LogParser()
    a = p.parse("Pod foo-12345 failed on node worker-1")
    b = p.parse("Pod bar-67890 failed on node worker-2")
    # Difieren solo en tokens dinámicos → misma plantilla / cluster
    assert a.cluster_id == b.cluster_id


@pytest.mark.unit
def test_different_logs_get_different_clusters():
    p = LogParser()
    a = p.parse("Successfully pulled image nginx")
    b = p.parse("Liveness probe failed with HTTP 500")
    assert a.cluster_id != b.cluster_id


@pytest.mark.unit
def test_numbers_and_ips_are_masked_in_template():
    p = LogParser()
    out = p.parse("Connection refused to 10.0.0.5:8080 after 42 retries")
    # La plantilla abstrae IP y números: no debe contener los valores literales
    assert "10.0.0.5" not in out.template
    assert "42" not in out.template


@pytest.mark.unit
def test_cluster_count_and_templates_grow():
    p = LogParser()
    p.parse("Started container A")
    p.parse("Killing container B")
    assert p.cluster_count >= 2
    templates = p.get_templates()
    assert len(templates) == p.cluster_count
    assert all(isinstance(t, str) for t in templates.values())
