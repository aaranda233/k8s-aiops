"""
Notificador de incidentes por email.

Envía emails con el contexto completo del incidente.
Para Level 2 incluye links de aprobación/rechazo con token único.
"""

import secrets
import smtplib
import time
from dataclasses import dataclass, field
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


@dataclass
class PendingApproval:
    token: str
    incident_id: str
    kubectl_cmd: str
    created_at: float = field(default_factory=time.time)
    response: str | None = None  # "approved" | "rejected"


class Notifier:
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
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_pass = smtp_pass
        self.from_addr = from_addr
        self.to_addr = to_addr
        self.webhook_base_url = webhook_base_url.rstrip("/")

        # Token store compartido con el web server
        self._pending: dict[str, PendingApproval] = {}

    def register_approval_store(self, store: dict) -> None:
        """Conecta con el dict del web server para compartir tokens."""
        self._pending = store

    def notify_level1_executed(
        self,
        incident_id: str,
        namespaces: set[str],
        root_cause: str,
        kubectl_cmd: str,
        execution_output: str,
        investigation_steps: list[str],
    ) -> None:
        subject = f"[K8s-AIOps] ✅ Incidente auto-resuelto — {', '.join(namespaces)}"
        body = self._build_html(
            incident_id=incident_id,
            namespaces=namespaces,
            root_cause=root_cause,
            kubectl_cmd=kubectl_cmd,
            execution_output=execution_output,
            investigation_steps=investigation_steps,
            level=1,
        )
        self._send(subject, body)

    def notify_level2_pending(
        self,
        incident_id: str,
        namespaces: set[str],
        root_cause: str,
        kubectl_cmd: str,
        risk_reason: str,
        investigation_steps: list[str],
    ) -> str:
        """Envía email con botones de aprobación. Devuelve el token."""
        token = secrets.token_urlsafe(16)
        self._pending[token] = PendingApproval(
            token=token,
            incident_id=incident_id,
            kubectl_cmd=kubectl_cmd,
        )

        approve_url = f"{self.webhook_base_url}/remediation/approve/{token}"
        reject_url = f"{self.webhook_base_url}/remediation/reject/{token}"

        subject = f"[K8s-AIOps] ⚠️ Aprobación requerida — {', '.join(namespaces)}"
        body = self._build_html(
            incident_id=incident_id,
            namespaces=namespaces,
            root_cause=root_cause,
            kubectl_cmd=kubectl_cmd,
            execution_output="",
            investigation_steps=investigation_steps,
            level=2,
            risk_reason=risk_reason,
            approve_url=approve_url,
            reject_url=reject_url,
        )
        self._send(subject, body)
        return token

    def notify_level3(
        self,
        incident_id: str,
        namespaces: set[str],
        root_cause: str,
        kubectl_cmd: str,
        investigation_steps: list[str],
    ) -> None:
        subject = f"[K8s-AIOps] 🚨 Acción manual requerida — {', '.join(namespaces)}"
        body = self._build_html(
            incident_id=incident_id,
            namespaces=namespaces,
            root_cause=root_cause,
            kubectl_cmd=kubectl_cmd,
            execution_output="",
            investigation_steps=investigation_steps,
            level=3,
        )
        self._send(subject, body)

    def notify_circuit_breaker(
        self,
        incident_id: str,
        namespaces: set[str],
        attempts: int,
        root_cause: str,
    ) -> None:
        subject = f"[K8s-AIOps] 🔴 Circuit breaker — {', '.join(namespaces)}"
        body = f"""
        <h2>Circuit Breaker activado</h2>
        <p>El agente ha detectado <strong>{attempts} intentos fallidos</strong>
        para la misma anomalía en los últimos 10 minutos.</p>
        <p><strong>Namespace:</strong> {', '.join(namespaces)}</p>
        <p><strong>Última causa raíz:</strong> {root_cause}</p>
        <p>Se requiere intervención manual. El agente no tomará más acciones automáticas
        para esta anomalía hasta que se resuelva o expire la ventana de 10 minutos.</p>
        <p><small>Incident ID: {incident_id}</small></p>
        """
        self._send(subject, body)

    def get_response(self, token: str) -> str | None:
        """Consulta si el token fue aprobado/rechazado."""
        entry = self._pending.get(token)
        return entry.response if entry else None

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
            # Log pero no propagar — el email nunca debe romper el pipeline
            import logging
            logging.getLogger(__name__).error("Error enviando email: %s", e)

    def _build_html(
        self,
        incident_id: str,
        namespaces: set[str],
        root_cause: str,
        kubectl_cmd: str,
        execution_output: str,
        investigation_steps: list[str],
        level: int,
        risk_reason: str = "",
        approve_url: str = "",
        reject_url: str = "",
    ) -> str:
        steps_html = "".join(f"<li><code>{s}</code></li>" for s in investigation_steps)
        level_badge = {
            1: '<span style="background:#16a34a;color:#fff;padding:2px 8px;border-radius:4px">Level 1 — Auto-ejecutado</span>',
            2: '<span style="background:#d97706;color:#fff;padding:2px 8px;border-radius:4px">Level 2 — Requiere aprobación</span>',
            3: '<span style="background:#dc2626;color:#fff;padding:2px 8px;border-radius:4px">Level 3 — Solo manual</span>',
        }.get(level, "")

        approval_section = ""
        if level == 2:
            approval_section = f"""
            <div style="margin:24px 0;padding:16px;background:#fffbeb;border:1px solid #fbbf24;border-radius:8px">
                <p><strong>Razón de escalación:</strong> {risk_reason}</p>
                <p style="margin-top:12px">
                    <a href="{approve_url}" style="background:#16a34a;color:#fff;padding:10px 20px;border-radius:6px;text-decoration:none;margin-right:12px;font-weight:600">
                        ✅ APROBAR
                    </a>
                    <a href="{reject_url}" style="background:#dc2626;color:#fff;padding:10px 20px;border-radius:6px;text-decoration:none;font-weight:600">
                        ❌ RECHAZAR
                    </a>
                </p>
                <p style="font-size:12px;color:#666;margin-top:8px">
                    Sin respuesta en 30 minutos → el agente descartará la acción.
                </p>
            </div>"""
        elif level == 3:
            approval_section = f"""
            <div style="margin:24px 0;padding:16px;background:#fef2f2;border:1px solid #fca5a5;border-radius:8px">
                <p><strong>⚠️ Acción destructiva — requiere ejecución manual:</strong></p>
                <code style="display:block;background:#1e1e1e;color:#d4d4d4;padding:12px;border-radius:4px;margin-top:8px">{kubectl_cmd}</code>
            </div>"""

        execution_section = ""
        if execution_output:
            execution_section = f"""
            <h3>Output de la ejecución</h3>
            <pre style="background:#1e1e1e;color:#d4d4d4;padding:12px;border-radius:4px;font-size:12px;overflow:auto">{execution_output[:1000]}</pre>"""

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="font-family:system-ui,sans-serif;max-width:700px;margin:0 auto;padding:24px;color:#1c1917">
    <h2 style="margin-bottom:4px">K8s-AIOps — Incidente Detectado</h2>
    <p style="color:#666;margin-top:0">{level_badge} &nbsp; <small>ID: {incident_id}</small></p>

    <table style="width:100%;border-collapse:collapse;margin:16px 0">
        <tr><td style="padding:6px 0;color:#666;width:140px">Namespace(s)</td>
            <td><strong>{', '.join(sorted(namespaces))}</strong></td></tr>
    </table>

    <h3>Investigación realizada</h3>
    <ul style="background:#f5f5f4;padding:12px 12px 12px 28px;border-radius:6px">{steps_html}</ul>

    <h3>Diagnóstico</h3>
    <div style="background:#eff6ff;border-left:4px solid #2563eb;padding:12px 16px;border-radius:0 6px 6px 0">
        {root_cause}
    </div>

    <h3>Acción propuesta</h3>
    <code style="display:block;background:#1e1e1e;color:#d4d4d4;padding:12px;border-radius:4px">{kubectl_cmd}</code>

    {approval_section}
    {execution_section}

    <hr style="margin-top:32px;border:none;border-top:1px solid #e5e2da">
    <p style="font-size:12px;color:#999">K8s-AIOps · {incident_id}</p>
</body></html>"""
