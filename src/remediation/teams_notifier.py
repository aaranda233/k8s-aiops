"""
Notificador de avisos para Microsoft Teams.

Publica una Adaptive Card de AVISO en el canal de ops vía Incoming Webhook.
La tarjeta NO lleva botones de aprobación: solo informa y ofrece un botón
"Ver en consola" (Action.OpenUrl) que abre /incidents/{id} en la consola web,
donde el operador revisa, aprueba/rechaza, edita o chatea con autenticación.

Este diseño separa el aviso (Teams) del control (web): Teams empuja, la web actúa.
"""

import logging

import httpx

from src.remediation.base_notifier import (
    KIND_APPROVAL,
    KIND_CIRCUIT,
    KIND_MANUAL,
    KIND_RESOLVED,
    BaseNotifier,
)

logger = logging.getLogger(__name__)

_COLOR = {
    KIND_RESOLVED: "good",
    KIND_APPROVAL: "warning",
    KIND_MANUAL: "attention",
    KIND_CIRCUIT: "attention",
}


class TeamsNotifier(BaseNotifier):
    def __init__(self, webhook_url: str, webhook_base_url: str = "http://localhost:8000"):
        super().__init__(webhook_base_url=webhook_base_url)
        self.webhook_url = webhook_url

    def notify(self, incident, kind: str) -> None:
        self._post(self._build_card(incident, kind))

    def _build_card(self, incident, kind: str) -> dict:
        namespaces = ", ".join(sorted(incident.namespaces))
        body = [
            {"type": "TextBlock", "text": f"K8s-AIOps — {self.title(kind)}",
             "weight": "Bolder", "size": "Large", "color": _COLOR.get(kind, "default")},
            {"type": "FactSet", "facts": [
                {"title": "Namespace(s)", "value": namespaces},
                {"title": "Incident", "value": incident.id},
                {"title": "Score", "value": f"{incident.score:.3f}"},
                {"title": "Riesgo", "value": f"Level {incident.risk_level} ({incident.risk_label})"},
            ]},
            {"type": "TextBlock", "text": "**Diagnóstico**", "weight": "Bolder", "spacing": "Medium"},
            {"type": "TextBlock", "text": incident.root_cause, "wrap": True},
            {"type": "TextBlock", "text": "**Acción propuesta**", "weight": "Bolder", "spacing": "Medium"},
            {"type": "TextBlock", "text": f"`{incident.kubectl_cmd}`", "wrap": True, "fontType": "Monospace"},
        ]

        if kind == KIND_APPROVAL:
            body.append({"type": "TextBlock",
                         "text": "Requiere tu decisión en la consola. Sin respuesta en 30 min → se descarta.",
                         "wrap": True, "isSubtle": True, "spacing": "Small"})

        card = {
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": body,
            "actions": [
                {"type": "Action.OpenUrl", "title": "🔎 Ver en consola",
                 "url": self.console_link(incident.id)},
            ],
        }
        return card

    def _post(self, card: dict) -> None:
        payload = {
            "type": "message",
            "attachments": [{
                "contentType": "application/vnd.microsoft.card.adaptive",
                "content": card,
            }],
        }
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(self.webhook_url, json=payload)
                resp.raise_for_status()
        except Exception as e:
            logger.error("Error enviando a Teams: %s", e)
