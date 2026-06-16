"""
Tests de la API web (web/server.py) con FastAPI TestClient.

Cubre los endpoints que no dependen de un cluster vivo: salud, listado y
detalle de incidentes, aprobación/rechazo, y la página principal. El hilo del
pipeline arranca en startup pero falla de forma controlada sin cluster.
"""

import pytest
from fastapi.testclient import TestClient

from web import server


@pytest.fixture(scope="module")
def client():
    with TestClient(server.app) as c:
        yield c


@pytest.mark.unit
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


@pytest.mark.unit
def test_ready_returns_known_status(client):
    r = client.get("/ready")
    assert r.status_code in (200, 503)


@pytest.mark.unit
def test_list_incidents_returns_envelope(client):
    r = client.get("/api/incidents")
    assert r.status_code == 200
    assert "incidents" in r.json()
    assert isinstance(r.json()["incidents"], list)


@pytest.mark.unit
def test_get_unknown_incident_404(client):
    r = client.get("/api/incidents/INC-NOPE")
    assert r.status_code == 404


@pytest.mark.unit
def test_approve_unknown_incident_404(client):
    r = client.post("/api/incidents/INC-NOPE/approve")
    assert r.status_code == 404


@pytest.mark.unit
def test_reject_unknown_incident_404(client):
    r = client.post("/api/incidents/INC-NOPE/reject")
    assert r.status_code == 404


@pytest.mark.unit
def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


@pytest.mark.unit
@pytest.mark.parametrize("path", ["/incidents", "/chat", "/topology", "/security"])
def test_spa_pages_served(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


@pytest.mark.unit
def test_topology_api_returns_envelope(client):
    # Sin cluster, el endpoint captura el error y devuelve un envelope 200
    r = client.get("/api/topology")
    assert r.status_code == 200
    body = r.json()
    assert "nodes" in body and "links" in body


@pytest.mark.unit
def test_security_api_returns_envelope(client):
    r = client.get("/api/security")
    assert r.status_code == 200
    body = r.json()
    assert "findings" in body


@pytest.mark.unit
def test_chat_stream_empty_query_emits_error(client):
    r = client.get("/api/chat/stream", params={"q": "  "})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert "error" in r.text.lower()


@pytest.mark.unit
def test_incident_detail_page_served(client):
    r = client.get("/incidents/INC-ANY")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


@pytest.mark.unit
def test_reject_existing_incident(client):
    import time

    from src.remediation.incident_store import STATUS_PENDING, Incident
    inc = Incident(
        id="INC-WEBTEST2", created_at=time.time(), namespaces=["prod"], score=0.9,
        root_cause="test", kubectl_cmd="kubectl get pods", risk_level=2,
        risk_label="config", investigation=[], status=STATUS_PENDING,
    )
    server.incident_store.add(inc)
    r = client.post("/api/incidents/INC-WEBTEST2/reject")
    assert r.status_code == 200
    assert r.json()["status"] == "rejected"


@pytest.mark.unit
def test_demo_endpoint_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("AIOPS_DEMO", raising=False)
    r = client.post("/api/demo/incident", params={"mode": "human"})
    assert r.status_code == 403


@pytest.mark.unit
def test_demo_endpoint_enabled_triggers_remediation(client, monkeypatch):
    monkeypatch.setenv("AIOPS_DEMO", "true")
    captured = {}

    import src.remediation.auto_remediation as armod

    def fake_handle_async(self, scored, diag):
        captured["kubectl"] = diag.kubectl_command
        captured["shadow"] = self.shadow_mode
    monkeypatch.setattr(armod.AutoRemediation, "handle_async", fake_handle_async)

    r = client.post("/api/demo/incident", params={"mode": "human"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "human"
    assert "rollout restart" in body["kubectl"]
    assert captured["shadow"] is True   # human → modo sombra (espera aprobación)

    r = client.post("/api/demo/incident", params={"mode": "auto"})
    assert r.json()["mode"] == "auto"
    assert captured["shadow"] is False  # auto → ejecuta sin aprobación


@pytest.mark.unit
def test_approve_then_reject_existing_incident(client):
    """Inyecta un incidente en el store compartido y prueba el ciclo de decisión."""
    import time

    from src.remediation.incident_store import STATUS_PENDING, Incident
    inc = Incident(
        id="INC-WEBTEST1", created_at=time.time(), namespaces=["prod"], score=0.9,
        root_cause="test", kubectl_cmd="kubectl get pods", risk_level=2,
        risk_label="config", investigation=[], status=STATUS_PENDING,
    )
    server.incident_store.add(inc)

    r = client.get("/api/incidents/INC-WEBTEST1")
    assert r.status_code == 200
    assert r.json()["id"] == "INC-WEBTEST1"

    r = client.post("/api/incidents/INC-WEBTEST1/approve")
    assert r.status_code == 200
    assert r.json()["status"] == "approved"
