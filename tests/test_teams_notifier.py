"""Tests del notificador de Teams (solo-aviso, sin POST real)."""

import time
from unittest.mock import patch

import pytest

from src.remediation.base_notifier import KIND_APPROVAL, KIND_RESOLVED, CompositeNotifier, build_notifier
from src.remediation.incident_store import Incident
from src.remediation.teams_notifier import TeamsNotifier


def _incident():
    return Incident(
        id="INC-1", created_at=time.time(), namespaces=["prod"], score=0.94,
        root_cause="Disco lleno en node-1", kubectl_cmd="kubectl drain node-1",
        risk_level=3, risk_label="destructivo", investigation=["THOUGHT: revisar disco"],
    )


def _teams():
    return TeamsNotifier(webhook_url="https://outlook.office.com/webhook/test",
                         webhook_base_url="http://console:8000")


@pytest.mark.unit
def test_card_has_console_link_only():
    n = _teams()
    captured = {}
    with patch.object(n, "_post", side_effect=lambda c: captured.update(card=c)):
        n.notify(_incident(), KIND_APPROVAL)
    actions = captured["card"]["actions"]
    # Solo el botón "Ver en consola" — sin APROBAR/RECHAZAR en Teams
    assert len(actions) == 1
    assert actions[0]["url"] == "http://console:8000/incidents/INC-1"
    titles = [a["title"].lower() for a in actions]
    assert all("aprobar" not in t and "rechazar" not in t for t in titles)


@pytest.mark.unit
def test_card_contains_diagnosis_and_command():
    n = _teams()
    captured = {}
    with patch.object(n, "_post", side_effect=lambda c: captured.update(card=c)):
        n.notify(_incident(), KIND_RESOLVED)
    body_text = str(captured["card"]["body"])
    assert "Disco lleno en node-1" in body_text
    assert "kubectl drain node-1" in body_text


@pytest.mark.unit
def test_card_is_adaptive():
    n = _teams()
    captured = {}
    with patch.object(n, "_post", side_effect=lambda c: captured.update(card=c)):
        n.notify(_incident(), KIND_RESOLVED)
    assert captured["card"]["type"] == "AdaptiveCard"


@pytest.mark.unit
def test_post_failure_does_not_raise():
    import src.remediation.teams_notifier as tn
    n = _teams()
    with patch.object(tn.httpx, "Client", side_effect=Exception("network down")):
        n.notify(_incident(), KIND_RESOLVED)  # no debe lanzar


@pytest.mark.unit
def test_composite_fans_out_to_all_channels():
    n1, n2 = _teams(), _teams()
    composite = CompositeNotifier([n1, n2])
    calls = []
    with patch.object(n1, "_post", side_effect=lambda c: calls.append("n1")), \
         patch.object(n2, "_post", side_effect=lambda c: calls.append("n2")):
        composite.notify(_incident(), KIND_APPROVAL)
    assert calls == ["n1", "n2"]


@pytest.mark.unit
def test_composite_continues_if_one_channel_fails():
    n1, n2 = _teams(), _teams()
    composite = CompositeNotifier([n1, n2])
    calls = []
    with patch.object(n1, "_post", side_effect=Exception("teams down")), \
         patch.object(n2, "_post", side_effect=lambda c: calls.append("n2")):
        composite.notify(_incident(), KIND_APPROVAL)  # no debe lanzar
    assert calls == ["n2"]  # el segundo canal recibe el aviso pese al fallo del primero


@pytest.mark.unit
def test_build_notifier_channels():
    from config.settings import RemediationConfig

    teams = RemediationConfig()
    teams.notify_channel = "teams"
    teams.teams_webhook_url = "https://x"
    assert type(build_notifier(teams)).__name__ == "TeamsNotifier"

    both = RemediationConfig()
    both.notify_channel = "both"
    both.teams_webhook_url = "https://x"
    both.smtp_user = "u@x"
    both.notify_email = "s@x"
    assert type(build_notifier(both)).__name__ == "CompositeNotifier"

    none = RemediationConfig()
    none.notify_channel = "teams"
    none.teams_webhook_url = ""
    assert build_notifier(none) is None
