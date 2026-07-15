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
@pytest.mark.parametrize("path", ["/incidents", "/topology", "/security", "/graph"])
def test_spa_pages_served(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


@pytest.mark.unit
def test_graph_api_returns_envelope(client):
    r = client.get("/api/graph")
    assert r.status_code == 200
    body = r.json()
    assert "nodes" in body and isinstance(body["nodes"], list)
    assert "stats" in body
    assert {"nodes", "edges", "verified"} <= set(body["stats"])


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
def test_incident_detail_page_served(client):
    r = client.get("/incidents/INC-ANY")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


@pytest.mark.unit
def test_correct_writes_feedback(client, tmp_path, monkeypatch):
    """B) La corrección humana se guarda como feedback (señal de aprendizaje)."""
    import time

    from src.remediation.incident_store import STATUS_FAILED, Incident
    fb = tmp_path / "feedback.jsonl"
    monkeypatch.setenv("AIOPS_FEEDBACK_FILE", str(fb))
    inc = Incident(
        id="INC-CORR1", created_at=time.time(), namespaces=["prod"], score=0.9,
        root_cause="diagnóstico flojo", kubectl_cmd="kubectl get pods", risk_level=1,
        risk_label="reversible", status=STATUS_FAILED, prompt_user="eventos del incidente",
    )
    server.incident_store.add(inc)
    r = client.post("/api/incidents/INC-CORR1/correct",
                    json={"root_cause": "era OOMKilled", "kubectl": "kubectl set resources deploy/x"})
    assert r.status_code == 200
    assert r.json()["status"] == "corrected"
    # se escribió un ejemplo de feedback con la corrección
    lines = fb.read_text().strip().splitlines()
    assert any("era OOMKilled" in line for line in lines)


@pytest.mark.unit
def test_correct_empty_is_rejected(client):
    import time

    from src.remediation.incident_store import Incident
    inc = Incident(id="INC-CORR2", created_at=time.time(), namespaces=["p"], score=0.9,
                   root_cause="x", kubectl_cmd="k", risk_level=1, risk_label="r",
                   prompt_user="ev")
    server.incident_store.add(inc)
    r = client.post("/api/incidents/INC-CORR2/correct", json={"root_cause": "", "kubectl": ""})
    assert r.status_code == 400


@pytest.mark.unit
def test_correct_unknown_incident_404(client):
    r = client.post("/api/incidents/INC-NOPE/correct", json={"root_cause": "x"})
    assert r.status_code == 404


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
def test_demo_enabled_flag_reflects_env(client, monkeypatch):
    monkeypatch.delenv("AIOPS_DEMO", raising=False)
    assert client.get("/api/demo/enabled").json() == {"enabled": False}
    monkeypatch.setenv("AIOPS_DEMO", "true")
    assert client.get("/api/demo/enabled").json() == {"enabled": True}


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


@pytest.mark.unit
def test_graph_teach_rejects_unsafe_step(client):
    """Un paso con verbo destructivo debe rechazarse (400) antes de persistir."""
    r = client.post("/api/graph/teach", json={
        "intent": "prueba-insegura",
        "steps": [{"type": "command", "action": "kubectl delete pod x -n web"}],
    })
    assert r.status_code == 400


@pytest.mark.unit
def test_graph_teach_requires_intent_and_steps(client):
    assert client.post("/api/graph/teach", json={"steps": []}).status_code == 400
    assert client.post("/api/graph/teach", json={"intent": "x", "steps": []}).status_code == 400


@pytest.mark.unit
def test_graph_teach_accepts_valid_plan(client, monkeypatch):
    """Un plan válido se persiste vía add_taught y prefija el intent con human:."""
    captured = {}

    class _FakeGraph:
        def add_taught(self, intent, steps, **kw):
            captured["intent"] = intent
            captured["n"] = len(steps)

    monkeypatch.setattr("src.remediation.remediation_graph.get_graph", lambda: _FakeGraph())
    r = client.post("/api/graph/teach", json={
        "intent": "oom-replica",
        "steps": [
            {"type": "investigate", "action": "kubectl describe pod {pod} -n {ns}"},
            {"type": "command", "action": "kubectl rollout restart deployment/{workload} -n {ns}"},
        ],
    })
    assert r.status_code == 200
    assert r.json()["status"] == "taught"
    assert captured["intent"] == "human:oom-replica"
    assert captured["n"] == 2


@pytest.mark.unit
def test_graph_draft_503_when_escalation_disabled(client, monkeypatch):
    monkeypatch.delenv("ESCALATION_BACKEND", raising=False)
    r = client.post("/api/graph/draft", json={"root_cause": "algo raro"})
    assert r.status_code == 503


@pytest.mark.unit
def test_chat_remediation_requires_message(client):
    assert client.post("/api/chat/remediation", json={"incident_id": "x"}).status_code == 400


@pytest.mark.unit
def test_chat_remediation_503_when_disabled(client, monkeypatch):
    monkeypatch.delenv("ESCALATION_BACKEND", raising=False)
    r = client.post("/api/chat/remediation", json={"incident_id": "x", "message": "hola"})
    assert r.status_code == 503


@pytest.mark.unit
def test_chat_apply_incident_not_found(client):
    r = client.post("/api/chat/remediation/apply",
                    json={"incident_id": "nope", "steps": [
                        {"type": "investigate", "action": "kubectl get pods -n web"}]})
    assert r.status_code == 404


@pytest.mark.unit
def test_chat_apply_validates_and_attaches(client, monkeypatch):
    """Un plan con paso destructivo se rechaza; uno válido se adjunta al incidente."""
    import time

    from src.remediation.incident_store import Incident
    from web import server as srv

    inc = Incident(id="chat-test-1", created_at=time.time(), namespaces=["web"],
                   score=0.9, root_cause="CrashLoop", kubectl_cmd="kubectl get pods -n web",
                   risk_level=1, risk_label="reversible")
    srv.incident_store.add(inc)

    # destructivo → 400
    bad = client.post("/api/chat/remediation/apply", json={
        "incident_id": "chat-test-1",
        "steps": [{"type": "command", "action": "kubectl delete pod x -n web"}]})
    assert bad.status_code == 400

    # válido → 200 y queda adjunto al incidente
    ok = client.post("/api/chat/remediation/apply", json={
        "incident_id": "chat-test-1",
        "steps": [
            {"type": "investigate", "action": "kubectl describe pod api -n web"},
            {"type": "command", "action": "kubectl rollout restart deployment/api -n web"}]})
    assert ok.status_code == 200
    assert ok.json()["steps"] == 2
    plan = srv.incident_store.get("chat-test-1").remediation_plan
    assert len(plan) == 2 and plan[0]["action_type"] == "investigate"
