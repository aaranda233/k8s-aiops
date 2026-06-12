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
        sample = w.raw_logs[-self.max_logs:]
        logs_text = "\n".join(f"  {l}" for l in sample)

        user_msg = (
            f"Anomaly Score: {scored_window.score:.3f}\n"
            f"Namespaces affected: {', '.join(w.namespaces)}\n"
            f"Window: t={w.start_time:.0f}s – t={w.end_time:.0f}s\n"
            f"Total events: {w.log_count} | Distinct templates: {w.template_count}\n"
            f"Event sample (last {len(sample)}):\n{logs_text}"
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
        root_cause, kubectl_cmd = self._parse(text)

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
        root_cause = "Could not parse root cause."
        kubectl_cmd = "kubectl get events --all-namespaces --sort-by='.lastTimestamp'"
        for line in text.splitlines():
            if line.startswith("ROOT CAUSE:"):
                root_cause = line.removeprefix("ROOT CAUSE:").strip()
            elif line.startswith("KUBECTL:"):
                kubectl_cmd = line.removeprefix("KUBECTL:").strip()
        return root_cause, kubectl_cmd

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as c:
                resp = c.get(f"{self.host}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                return any(self.model in m for m in models)
        except Exception:
            return False
