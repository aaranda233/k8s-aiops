"""
Notificador de incidentes para Microsoft Teams.

Publica Adaptive Cards en un Incoming Webhook de Teams (canal de ops).
Para Level 2, los botones usan Action.OpenUrl apuntando a los endpoints
de aprobación del web server (/remediation/approve|reject/{token}),
reutilizando el mismo mecanismo de tokens que el email.

Nota de diseño: con un Incoming Webhook, los botones abren una pestaña del
navegador hacia el endpoint de aprobación (Action.OpenUrl). Para confirmación
inline (la tarjeta se actualiza en el propio Teams con Action.Execute) haría
falta registrar un Bot del Bot Framework — fuera del alcance actual. OpenUrl
es la vía ligera que funciona solo con la URL del webhook.
"""

import logging

import httpx

from src.remediation.base_notifier import BaseNotifier

logger = logging.getLogger(__name__)

_COLOR = {1: "good", 2: "warning", 3: "attention"}
_TITLE = {
    1: "✅ Incidente auto-resuelto",
    2: "⚠️ Aprobación requerida",
    3: "🚨 Acción manual requerida",
}


class TeamsNotifier(BaseNotifier):
    def __init__(self, webhook_url: str, webhook_base_url: str = "http://localhost:8000"):
        super().__init__(webhook_base_url=webhook_base_url)
        self.webhook_url = webhook_url

    def notify_level1_executed(self, incident_id, namespaces, root_cause, kubectl_cmd, execution_output, investigation_steps) -> None:
        card = self._build_card(
            level=1, incident_id=incident_id, namespaces=namespaces,
            root_cause=root_cause, kubectl_cmd=kubectl_cmd,
            investigation_steps=investigation_steps, execution_output=execution_output,
        )
        self._post(card)

    def notify_level2_pending(self, incident_id, namespaces, root_cause, kubectl_cmd, risk_reason, investigation_steps) -> str:
        token = self._make_token(incident_id, kubectl_cmd)
        self._notify_level2_with_token(
            token, incident_id, namespaces, root_cause, kubectl_cmd, risk_reason, investigation_steps
        )
        return token

    def _notify_level2_with_token(self, token, incident_id, namespaces, root_cause, kubectl_cmd, risk_reason, investigation_steps) -> None:
        approve_url, reject_url = self._approval_urls(token)
        card = self._build_card(
            level=2, incident_id=incident_id, namespaces=namespaces,
            root_cause=root_cause, kubectl_cmd=kubectl_cmd,
            investigation_steps=investigation_steps, risk_reason=risk_reason,
            approve_url=approve_url, reject_url=reject_url,
        )
        self._post(card)

    def notify_level3(self, incident_id, namespaces, root_cause, kubectl_cmd, investigation_steps) -> None:
        card = self._build_card(
            level=3, incident_id=incident_id, namespaces=namespaces,
            root_cause=root_cause, kubectl_cmd=kubectl_cmd,
            investigation_steps=investigation_steps,
        )
        self._post(card)

    def notify_circuit_breaker(self, incident_id, namespaces, attempts, root_cause) -> None:
        card = {
            "type": "AdaptiveCard",
            "version": "1.4",
            "body": [
                {"type": "TextBlock", "text": "🔴 Circuit Breaker activado", "weight": "Bolder", "size": "Large", "color": "attention"},
                {"type": "TextBlock", "text": f"{attempts} intentos fallidos para la misma anomalía en 10 minutos. Se requiere intervención manual.", "wrap": True},
                {"type": "FactSet", "facts": [
                    {"title": "Namespace(s)", "value": ", ".join(sorted(namespaces))},
                    {"title": "Causa raíz", "value": root_cause[:200]},
                    {"title": "Incident", "value": incident_id},
                ]},
            ],
        }
        self._post(card)

    def _build_card(
        self, level, incident_id, namespaces, root_cause, kubectl_cmd,
        investigation_steps, risk_reason="", approve_url="", reject_url="", execution_output="",
    ) -> dict:
        body = [
            {"type": "TextBlock", "text": f"K8s-AIOps — {_TITLE.get(level, 'Incidente')}",
             "weight": "Bolder", "size": "Large", "color": _COLOR.get(level, "default")},
            {"type": "FactSet", "facts": [
                {"title": "Namespace(s)", "value": ", ".join(sorted(namespaces))},
                {"title": "Incident", "value": incident_id},
            ]},
        ]

        if investigation_steps:
            steps_text = "\n".join(f"- {s}" for s in investigation_steps)
            body.append({"type": "TextBlock", "text": "**Investigación**", "weight": "Bolder", "spacing": "Medium"})
            body.append({"type": "TextBlock", "text": steps_text, "wrap": True, "isSubtle": True})

        body.append({"type": "TextBlock", "text": "**Diagnóstico**", "weight": "Bolder", "spacing": "Medium"})
        body.append({"type": "TextBlock", "text": root_cause, "wrap": True})
        body.append({"type": "TextBlock", "text": "**Acción propuesta**", "weight": "Bolder", "spacing": "Medium"})
        body.append({"type": "TextBlock", "text": f"`{kubectl_cmd}`", "wrap": True, "fontType": "Monospace"})

        if level == 3:
            body.append({"type": "TextBlock", "text": "⚠️ Acción destructiva — ejecútala manualmente tras revisar.",
                         "wrap": True, "color": "attention", "spacing": "Medium"})
        if execution_output:
            body.append({"type": "TextBlock", "text": f"```\n{execution_output[:500]}\n```", "wrap": True, "fontType": "Monospace", "isSubtle": True})

        card = {"type": "AdaptiveCard", "version": "1.4", "body": body}

        if level == 2:
            if risk_reason:
                body.append({"type": "TextBlock", "text": f"Razón de escalación: {risk_reason}", "wrap": True, "isSubtle": True, "spacing": "Small"})
            body.append({"type": "TextBlock", "text": "Sin respuesta en 30 min → la acción se descarta.", "wrap": True, "size": "Small", "isSubtle": True})
            card["actions"] = [
                {"type": "Action.OpenUrl", "title": "✅ APROBAR", "url": approve_url},
                {"type": "Action.OpenUrl", "title": "❌ RECHAZAR", "url": reject_url},
            ]

        return card

    def _post(self, card: dict) -> None:
        # Formato de Incoming Webhook: la Adaptive Card va envuelta en un attachment
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
            # Un fallo de Teams nunca debe romper el pipeline
            logger.error("Error enviando a Teams: %s", e)
