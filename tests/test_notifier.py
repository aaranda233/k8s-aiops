"""Tests del notificador (sin envío real de SMTP)."""

from unittest.mock import patch

import pytest

from src.remediation.notifier import Notifier, PendingApproval


def _notifier():
    n = Notifier(
        smtp_host="smtp.test", smtp_port=587,
        smtp_user="u", smtp_pass="p",
        from_addr="from@test", to_addr="to@test",
        webhook_base_url="http://localhost:8000",
    )
    return n


@pytest.mark.unit
def test_level2_creates_token_and_pending():
    n = _notifier()
    with patch.object(n, "_send"):
        token = n.notify_level2_pending(
            "INC-1", {"prod"}, "OOMKilled", "kubectl patch deployment x",
            "config change", ["THOUGHT: ..."],
        )
    assert token in n._pending
    assert isinstance(n._pending[token], PendingApproval)
    assert n.get_response(token) is None  # aún sin responder


@pytest.mark.unit
def test_approval_store_shared():
    n = _notifier()
    external_store = {}
    n.register_approval_store(external_store)
    with patch.object(n, "_send"):
        token = n.notify_level2_pending(
            "INC-1", {"prod"}, "rc", "kubectl patch x", "reason", [],
        )
    # El token debe estar en el store externo (el del web server)
    assert token in external_store


@pytest.mark.unit
def test_get_response_after_approval():
    n = _notifier()
    with patch.object(n, "_send"):
        token = n.notify_level2_pending("INC-1", {"prod"}, "rc", "kubectl patch x", "r", [])
    n._pending[token].response = "approved"
    assert n.get_response(token) == "approved"


@pytest.mark.unit
def test_email_body_contains_diagnosis_and_command():
    n = _notifier()
    sent = {}
    def capture(subject, body):
        sent["subject"] = subject
        sent["body"] = body
    with patch.object(n, "_send", side_effect=capture):
        n.notify_level1_executed(
            "INC-42", {"prod"}, "Memory pressure detectada",
            "kubectl rollout restart deployment/x -n prod",
            "deployment.apps/x restarted", ["THOUGHT: revisar nodo"],
        )
    assert "Memory pressure detectada" in sent["body"]
    assert "kubectl rollout restart" in sent["body"]
    assert "INC-42" in sent["body"]


@pytest.mark.unit
def test_level2_email_has_approve_reject_links():
    n = _notifier()
    sent = {}
    with patch.object(n, "_send", side_effect=lambda s, b: sent.update(body=b)):
        token = n.notify_level2_pending("INC-1", {"prod"}, "rc", "kubectl patch x", "r", [])
    assert f"/remediation/approve/{token}" in sent["body"]
    assert f"/remediation/reject/{token}" in sent["body"]


@pytest.mark.unit
def test_send_failure_does_not_raise():
    """Un fallo de SMTP nunca debe romper el pipeline."""
    n = _notifier()
    # smtp.test no resuelve → _send captura la excepción internamente
    n.notify_circuit_breaker("INC-1", {"prod"}, 3, "OOMKilled")  # no debe lanzar
