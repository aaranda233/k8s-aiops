"""
Notificador de avisos por email (SMTP) — canal fallback.

Igual que TeamsNotifier, solo AVISA: envía un email con el resumen del
incidente y un enlace "Ver en consola" hacia /incidents/{id}. La decisión
humana ocurre en la consola web, no en el email.
"""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from src.remediation.base_notifier import KIND_APPROVAL, BaseNotifier

logger = logging.getLogger(__name__)

__all__ = ["EmailNotifier", "Notifier"]


class EmailNotifier(BaseNotifier):
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_pass: str,
        from_addr: str,
        to_addr: str,
        webhook_base_url: str = "http://localhost:8000",
    ):
        super().__init__(webhook_base_url=webhook_base_url)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_pass = smtp_pass
        self.from_addr = from_addr
        self.to_addr = to_addr

    def notify(self, incident, kind: str) -> None:
        ns = ", ".join(sorted(incident.namespaces))
        subject = f"[K8s-AIOps] {self.title(kind)} — {ns}"
        self._send(subject, self._build_html(incident, kind))

    def _build_html(self, incident, kind: str) -> str:
        steps_html = "".join(f"<li><code>{s}</code></li>" for s in incident.investigation)
        link = self.console_link(incident.id)
        approval_note = ""
        if kind == KIND_APPROVAL:
            approval_note = (
                '<p style="font-size:13px;color:#92400e">Requiere tu decisión en la consola. '
                "Sin respuesta en 30 min → la acción se descarta.</p>"
            )
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="font-family:system-ui,sans-serif;max-width:700px;margin:0 auto;padding:24px;color:#1c1917">
    <h2 style="margin-bottom:4px">K8s-AIOps — {self.title(kind)}</h2>
    <p style="color:#666;margin-top:0"><small>ID: {incident.id} · score {incident.score:.3f} ·
       Level {incident.risk_level} ({incident.risk_label})</small></p>

    <table style="width:100%;border-collapse:collapse;margin:16px 0">
        <tr><td style="padding:6px 0;color:#666;width:140px">Namespace(s)</td>
            <td><strong>{', '.join(sorted(incident.namespaces))}</strong></td></tr>
    </table>

    <h3>Investigación</h3>
    <ul style="background:#f5f5f4;padding:12px 12px 12px 28px;border-radius:6px">{steps_html}</ul>

    <h3>Diagnóstico</h3>
    <div style="background:#eff6ff;border-left:4px solid #2563eb;padding:12px 16px;border-radius:0 6px 6px 0">
        {incident.root_cause}
    </div>

    <h3>Acción propuesta</h3>
    <code style="display:block;background:#1e1e1e;color:#d4d4d4;padding:12px;border-radius:4px">{incident.kubectl_cmd}</code>

    {approval_note}

    <p style="margin:24px 0">
        <a href="{link}" style="background:#2563eb;color:#fff;padding:10px 20px;border-radius:6px;text-decoration:none;font-weight:600">
            🔎 Ver en consola
        </a>
    </p>

    <hr style="margin-top:32px;border:none;border-top:1px solid #e5e2da">
    <p style="font-size:12px;color:#999">K8s-AIOps · {incident.id}</p>
</body></html>"""

    def _send(self, subject: str, html_body: str) -> None:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = self.to_addr
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as s:
                s.ehlo()
                s.starttls()
                s.login(self.smtp_user, self.smtp_pass)
                s.sendmail(self.from_addr, [self.to_addr], msg.as_string())
        except Exception as e:
            logger.error("Error enviando email: %s", e)


# Alias de compatibilidad
Notifier = EmailNotifier
