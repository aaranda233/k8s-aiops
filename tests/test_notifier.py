"""Tests del notificador de email (solo-aviso, sin SMTP real)."""

import time
from unittest.mock import patch

import pytest

from src.remediation.base_notifier import KIND_APPROVAL, KIND_RESOLVED
from src.remediation.incident_store import Incident
from src.remediation.notifier import EmailNotifier, Notifier


def _incident():
    return Incident(
        id="INC-42", created_at=time.time(), namespaces=["prod"], score=0.91,
        root_cause="Memory pressure en node-1", kubectl_cmd="kubectl rollout restart deployment/x -n prod",
        risk_level=1, risk_label="reversible", investigation=["THOUGHT: revisar nodo"],
    )


def _notifier():
    return EmailNotifier(
        smtp_host="smtp.test", smtp_port=587, smtp_user="u", smtp_pass="p",
        from_addr="from@test", to_addr="to@test", webhook_base_url="http://console:8000",
    )


@pytest.mark.unit
def test_alias_notifier_is_email():
    assert Notifier is EmailNotifier


@pytest.mark.unit
def test_email_contains_diagnosis_and_console_link():
    n = _notifier()
    sent = {}
    with patch.object(n, "_send", side_effect=lambda subj, body: sent.update(subject=subj, body=body)):
        n.notify(_incident(), KIND_RESOLVED)
    assert "Memory pressure en node-1" in sent["body"]
    assert "kubectl rollout restart" in sent["body"]
    # Solo-aviso: enlace a la consola, NO botones de aprobación
    assert "http://console:8000/incidents/INC-42" in sent["body"]
    assert "/remediation/approve" not in sent["body"]


@pytest.mark.unit
def test_approval_email_has_console_link_not_buttons():
    n = _notifier()
    sent = {}
    with patch.object(n, "_send", side_effect=lambda subj, body: sent.update(body=body)):
        n.notify(_incident(), KIND_APPROVAL)
    assert "/incidents/INC-42" in sent["body"]
    # La decisión se toma en la consola, no en el email
    assert "consola" in sent["body"].lower()


@pytest.mark.unit
def test_send_failure_does_not_raise():
    n = _notifier()  # smtp.test no resuelve
    n.notify(_incident(), KIND_RESOLVED)  # no debe lanzar
