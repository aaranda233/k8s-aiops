"""
Capa 3 — Root Cause Analysis via SLM local (Ollama).
Solo se activa de forma reactiva cuando score >= threshold.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    pass

_SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer (SRE) specialized in Kubernetes.
You receive raw Kubernetes events from a time window flagged as anomalous by an ML model.
Your task:
1. Identify the root cause in 2-3 sentences.
2. Propose ONE specific kubectl command to investigate or mitigate.

Output format (strict, no extra text):
ROOT CAUSE: <explanation>
KUBECTL: <exact command>"""

_DEFAULT_KUBECTL = "kubectl get events --all-namespaces --sort-by='.lastTimestamp'"
# Límite de contexto del experto (num_ctx=2048). Acotamos la muestra de eventos
# para que el prompt + la salida no superen la ventana: si se supera, Ollama
# trunca por la izquierda y se pierden el system prompt y el marcador assistant,
# produciendo salida inparseable. Líneas largas (stack traces/JSON) eran la causa.
_MAX_LINE_CHARS = 200
_MAX_SAMPLE_CHARS = 3500


def build_event_sample(raw_logs: list, max_logs: int = 40) -> tuple[str, int]:
    """Construye una muestra de eventos acotada (por línea y total) para el prompt."""
    sample = list(raw_logs)[-max_logs:]
    lines = []
    for entry in sample:
        s = str(entry)
        if len(s) > _MAX_LINE_CHARS:
            s = s[:_MAX_LINE_CHARS] + "…"
        lines.append(f"  {s}")
    text = "\n".join(lines)
    if len(text) > _MAX_SAMPLE_CHARS:
        text = text[-_MAX_SAMPLE_CHARS:]  # conservar lo más reciente
    return text, len(sample)


def parse_diagnosis(text: str) -> tuple[str, str]:
    """Extrae (root_cause, kubectl) de la salida del modelo de forma tolerante.

    Acepta variaciones de formato (markdown, mayúsculas, espacios). Si no encuentra
    el patrón estricto, usa el texto del modelo como causa (mejor que "no parseable")
    y un kubectl por defecto. Nunca devuelve "Could not parse" si el modelo dijo algo.
    """
    root_cause = None
    kubectl_cmd = None
    for raw in text.splitlines():
        line = raw.strip().strip("*").strip()
        low = line.lower()
        if root_cause is None and low.startswith("root cause"):
            root_cause = line.split(":", 1)[1].strip() if ":" in line else ""
        if kubectl_cmd is None and low.startswith("kubectl"):
            ki = low.find("kubectl ")  # salta un posible prefijo "KUBECTL: "
            if ki != -1:
                kubectl_cmd = line[ki:].strip().strip("`").strip()

    if not root_cause:
        # Fallback: usar el texto del modelo (sin la línea de comando) como causa.
        cleaned = [l.strip() for l in text.splitlines()
                   if l.strip() and "kubectl" not in l.lower()]
        root_cause = " ".join(cleaned)[:400] if cleaned else "Sin causa raíz determinable."
    if not kubectl_cmd:
        kubectl_cmd = _DEFAULT_KUBECTL
    return root_cause, kubectl_cmd


@dataclass
class DiagnosisResult:
    window_index: int
    anomaly_score: float
    namespaces: set[str]
    root_cause: str
    kubectl_command: str
    model_version: int
    # Campos ReAct (opcionales, compatibles con single-shot)
    confidence: str = "unknown"
    steps_taken: int = 1
    react_trace: list = field(default_factory=list)
    mode: str = "single_shot"


class OllamaRCA:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:1.5b",
        max_logs: int = 40,
        timeout: float = 120.0,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.max_logs = max_logs
        self.timeout = timeout

    def diagnose(self, scored_window) -> DiagnosisResult:
        w = scored_window.window
        logs_text, n_sample = build_event_sample(w.raw_logs, self.max_logs)

        user_msg = (
            f"Anomaly Score: {scored_window.score:.3f}\n"
            f"Namespaces affected: {', '.join(w.namespaces)}\n"
            f"Window: t={w.start_time:.0f}s – t={w.end_time:.0f}s\n"
            f"Total events: {w.log_count} | Distinct templates: {w.template_count}\n"
            f"Event sample (last {n_sample}):\n{logs_text}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300},
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()

        text = resp.json()["message"]["content"].strip()
        root_cause, kubectl_cmd = parse_diagnosis(text)

        return DiagnosisResult(
            window_index=w.index,
            anomaly_score=scored_window.score,
            namespaces=w.namespaces,
            root_cause=root_cause,
            kubectl_command=kubectl_cmd,
            model_version=scored_window.model_version,
        )

    @staticmethod
    def _parse(text: str) -> tuple[str, str]:
        return parse_diagnosis(text)

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as c:
                resp = c.get(f"{self.host}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                return any(self.model in m for m in models)
        except Exception:
            return False
