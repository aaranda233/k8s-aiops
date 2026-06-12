"""Tests del notificador de Teams (sin POST real al webhook)."""

from unittest.mock import patch

import pytest

from src.remediation.base_notifier import CompositeNotifier, build_notifier
from src.remediation.teams_notifier import TeamsNotifier


def _teams():
    return TeamsNotifier(webhook_url="https://outlook.office.com/webhook/test",
                         webhook_base_url="http://localhost:8000")


@pytest.mark.unit
def test_level2_creates_token_and_posts():
    n = _teams()
    with patch.object(n, "_post") as mock_post:
        token = n.notify_level2_pending(
            "INC-1", {"prod"}, "OOMKilled", "kubectl patch deployment x",
            "config change", ["THOUGHT: revisar nodo"],
        )
    assert token in n._pending
    mock_post.assert_called_once()
    card = mock_post.call_args.args[0]
    assert card["type"] == "AdaptiveCard"


@pytest.mark.unit
def test_level2_card_has_approve_reject_actions():
    n = _teams()
    captured = {}
    with patch.object(n, "_post", side_effect=lambda c: captured.update(card=c)):
        token = n.notify_level2_pending("INC-1", {"prod"}, "rc", "kubectl patch x", "r", [])
    actions = captured["card"]["actions"]
    urls = [a["url"] for a in actions]
    assert f"http://localhost:8000/remediation/approve/{token}" in urls
    assert f"http://localhost:8000/remediation/reject/{token}" in urls


@pytest.mark.unit
def test_level1_card_no_actions():
    n = _teams()
    captured = {}
    with patch.object(n, "_post", side_effect=lambda c: captured.update(card=c)):
        n.notify_level1_executed("INC-1", {"prod"}, "rc", "kubectl rollout restart x", "restarted", [])
    # Level 1 ya ejecutado → sin botones de aprobación
    assert "actions" not in captured["card"]


@pytest.mark.unit
def test_card_contains_diagnosis_and_command():
    n = _teams()
    captured = {}
    with patch.object(n, "_post", side_effect=lambda c: captured.update(card=c)):
        n.notify_level3("INC-9", {"prod"}, "Disco lleno en node-1", "kubectl drain node-1", [])
    body_text = str(captured["card"]["body"])
    assert "Disco lleno en node-1" in body_text
    assert "kubectl drain node-1" in body_text


@pytest.mark.unit
def test_post_failure_does_not_raise():
    """Un fallo de Teams nunca debe romper el pipeline."""
    import src.remediation.teams_notifier as tn
    n = _teams()
    with patch.object(tn.httpx, "Client", side_effect=Exception("network down")):
        n.notify_circuit_breaker("INC-1", {"prod"}, 3, "OOMKilled")  # no debe lanzar


@pytest.mark.unit
def test_composite_shares_single_token():
    """En modo 'both', ambos canales comparten el mismo token."""
    n1 = _teams()
    n2 = _teams()
    composite = CompositeNotifier([n1, n2])
    with patch.object(n1, "_post"), patch.object(n2, "_post"):
        token = composite.notify_level2_pending(
            "INC-1", {"prod"}, "rc", "kubectl patch x", "r", [],
        )
    # El token existe en el store compartido y es el mismo para ambos
    assert token in composite._pending
    assert composite.get_response(token) is None
