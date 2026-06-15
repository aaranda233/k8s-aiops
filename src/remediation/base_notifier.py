"""
Base común de notificadores — capa de AVISO (push).

Tras el nuevo diseño, las notificaciones solo avisan: "ha pasado algo,
entra a la consola". La decisión humana (aprobar/rechazar/editar/chat)
ocurre en la consola web, no en el canal de notificación. Esto centraliza
el control y la autenticación en un único sitio.

Cada aviso incluye un deep-link a /incidents/{id} de la consola.
"""

import logging

logger = logging.getLogger(__name__)

# Tipos de aviso
KIND_APPROVAL = "approval_needed"     # Level 2 — requiere decisión en la consola
KIND_EXECUTED = "executed"            # Level 1 — ejecutado automáticamente
KIND_RESOLVED = "resolved"            # verificado OK
KIND_FAILED = "failed"                # ejecución/verificación falló
KIND_MANUAL = "manual_required"       # Level 3 — acción manual
KIND_CIRCUIT = "circuit_breaker"      # bucle detectado

_KIND_TITLE = {
    KIND_APPROVAL: "⚠️ Aprobación requerida",
    KIND_EXECUTED: "🔧 Acción ejecutada",
    KIND_RESOLVED: "✅ Incidente resuelto",
    KIND_FAILED: "❌ Remediación fallida",
    KIND_MANUAL: "🚨 Acción manual requerida",
    KIND_CIRCUIT: "🔴 Circuit breaker activado",
}


class BaseNotifier:
    """Capa de aviso. Las subclases implementan notify()."""

    def __init__(self, webhook_base_url: str = "http://localhost:8000"):
        self.webhook_base_url = webhook_base_url.rstrip("/")

    def console_link(self, incident_id: str) -> str:
        return f"{self.webhook_base_url}/incidents/{incident_id}"

    def title(self, kind: str) -> str:
        return _KIND_TITLE.get(kind, "Incidente")

    def notify(self, incident, kind: str) -> None:
        """incident: objeto con .id, .namespaces, .root_cause, .kubectl_cmd, etc."""
        raise NotImplementedError


class CompositeNotifier(BaseNotifier):
    """Reenvía el aviso a varios canales (ej. Teams + email)."""

    def __init__(self, notifiers: list[BaseNotifier]):
        super().__init__()
        self._notifiers = notifiers

    def notify(self, incident, kind: str) -> None:
        for n in self._notifiers:
            try:
                n.notify(incident, kind)
            except Exception as e:
                logger.error("Notificador %s falló: %s", type(n).__name__, e)


def build_notifier(cfg) -> BaseNotifier | None:
    """Construye el notificador según RemediationConfig.notify_channel.

    teams → TeamsNotifier · email → EmailNotifier · both → CompositeNotifier
    Devuelve None si no hay canal configurado correctamente.
    """
    from src.remediation.notifier import EmailNotifier
    from src.remediation.teams_notifier import TeamsNotifier

    channel = (cfg.notify_channel or "none").lower()
    notifiers: list[BaseNotifier] = []

    if channel in ("teams", "both") and cfg.teams_webhook_url:
        notifiers.append(TeamsNotifier(
            webhook_url=cfg.teams_webhook_url,
            webhook_base_url=cfg.webhook_base_url,
        ))

    if channel in ("email", "both") and cfg.smtp_user and cfg.notify_email:
        notifiers.append(EmailNotifier(
            smtp_host=cfg.smtp_host, smtp_port=cfg.smtp_port,
            smtp_user=cfg.smtp_user, smtp_pass=cfg.smtp_pass,
            from_addr=cfg.smtp_from or cfg.smtp_user,
            to_addr=cfg.notify_email,
            webhook_base_url=cfg.webhook_base_url,
        ))

    if not notifiers:
        return None
    if len(notifiers) == 1:
        return notifiers[0]
    return CompositeNotifier(notifiers)
