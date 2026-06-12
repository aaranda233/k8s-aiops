"""
Base común de notificadores de remediación.

Define la interfaz que usa el orquestador (AutoRemediation) y la lógica
compartida de tokens de aprobación. Las implementaciones concretas
(EmailNotifier, TeamsNotifier) renderizan y entregan el mensaje.

Los botones de aprobación apuntan a los endpoints del web server
(/remediation/approve/{token} y /reject/{token}), comunes a todos los canales.
"""

import secrets
import time
from dataclasses import dataclass, field


@dataclass
class PendingApproval:
    token: str
    incident_id: str
    kubectl_cmd: str
    created_at: float = field(default_factory=time.time)
    response: str | None = None  # "approved" | "rejected"


class BaseNotifier:
    """Lógica compartida de tokens. Las subclases implementan notify_*."""

    def __init__(self, webhook_base_url: str = "http://localhost:8000"):
        self.webhook_base_url = webhook_base_url.rstrip("/")
        self._pending: dict[str, PendingApproval] = {}

    def register_approval_store(self, store: dict) -> None:
        """Conecta con el dict del web server para compartir tokens."""
        self._pending = store

    def get_response(self, token: str) -> str | None:
        entry = self._pending.get(token)
        return entry.response if entry else None

    def _make_token(self, incident_id: str, kubectl_cmd: str) -> str:
        token = secrets.token_urlsafe(16)
        self._pending[token] = PendingApproval(
            token=token, incident_id=incident_id, kubectl_cmd=kubectl_cmd,
        )
        return token

    def _approval_urls(self, token: str) -> tuple[str, str]:
        return (
            f"{self.webhook_base_url}/remediation/approve/{token}",
            f"{self.webhook_base_url}/remediation/reject/{token}",
        )

    # Interfaz que el orquestador espera — implementada por las subclases
    def notify_level1_executed(self, incident_id, namespaces, root_cause, kubectl_cmd, execution_output, investigation_steps) -> None:
        raise NotImplementedError

    def notify_level2_pending(self, incident_id, namespaces, root_cause, kubectl_cmd, risk_reason, investigation_steps) -> str:
        raise NotImplementedError

    def notify_level3(self, incident_id, namespaces, root_cause, kubectl_cmd, investigation_steps) -> None:
        raise NotImplementedError

    def notify_circuit_breaker(self, incident_id, namespaces, attempts, root_cause) -> None:
        raise NotImplementedError


class CompositeNotifier(BaseNotifier):
    """Reenvía a varios canales (ej. Teams + email). El token lo crea el primero."""

    def __init__(self, notifiers: list[BaseNotifier]):
        super().__init__()
        self._notifiers = notifiers

    def register_approval_store(self, store: dict) -> None:
        self._pending = store
        for n in self._notifiers:
            n.register_approval_store(store)

    def notify_level1_executed(self, *args) -> None:
        for n in self._notifiers:
            n.notify_level1_executed(*args)

    def notify_level2_pending(self, incident_id, namespaces, root_cause, kubectl_cmd, risk_reason, investigation_steps) -> str:
        # Un único token compartido; el primer canal lo crea, el resto lo reutiliza
        token = self._make_token(incident_id, kubectl_cmd)
        for n in self._notifiers:
            n._pending = self._pending
            n._notify_level2_with_token(
                token, incident_id, namespaces, root_cause, kubectl_cmd, risk_reason, investigation_steps
            )
        return token

    def notify_level3(self, *args) -> None:
        for n in self._notifiers:
            n.notify_level3(*args)

    def notify_circuit_breaker(self, *args) -> None:
        for n in self._notifiers:
            n.notify_circuit_breaker(*args)


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
