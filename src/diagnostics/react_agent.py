"""
Agente ReAct para Root Cause Analysis en Kubernetes.

Ciclo: THOUGHT → ACTION → OBSERVATION → THOUGHT → ... → FINAL
El LLM investiga el cluster paso a paso en lugar de hacer una sola llamada ciega.
"""

from dataclasses import dataclass

import httpx

from src.diagnostics.kubectl_toolbox import execute as kubectl_execute
from src.diagnostics.ollama_rca import (
    DiagnosisResult,
    ensure_meaningful_root_cause,
    rca_focus,
    rca_namespaces_line,
    window_event_sample,
)

_SYSTEM_PROMPT = """\
You are an expert SRE investigating a Kubernetes anomaly step by step.

Available read-only tools:
  kubectl describe <resource> <name> [-n <namespace>]
  kubectl get <resource> [-n <namespace>] [-o yaml]
  kubectl logs <pod> [-n <namespace>] [--previous] [--tail=<N>]
  kubectl top pod [<name>] [-n <namespace>]
  kubectl get events [-n <namespace>] [--sort-by='.lastTimestamp']

Each turn, output EXACTLY ONE of these two formats:

Format A — investigate further:
THOUGHT: <reasoning about what to check next>
ACTION: kubectl <command>

Format B — conclude (when you have sufficient evidence):
THOUGHT: <summary of what you found>
FINAL:
ROOT CAUSE: <2-3 sentence explanation>
KUBECTL: <one remediation or verification command>
CONFIDENCE: <low|medium|high>

Rules:
- Use exact resource names and namespaces visible in the context
- Stop as soon as you have clear evidence — do not over-investigate
- If a command errors, try a different angle
- Output ONLY the format above, nothing else"""


@dataclass
class TraceStep:
    step: int
    thought: str
    action: str | None
    observation: str | None
    is_final: bool = False


class ReActAgent:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:1.5b",
        max_logs: int = 40,
        timeout: float = 120.0,
        max_steps: int = 5,
        dry_run: bool = True,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.max_logs = max_logs
        self.timeout = timeout
        self.max_steps = max_steps
        self.dry_run = dry_run

    def diagnose(self, scored_window) -> DiagnosisResult:
        """Ejecuta el ciclo ReAct y devuelve un DiagnosisResult compatible con el pipeline."""
        w = scored_window.window

        culprit = getattr(scored_window, "culprit_namespace", "") or None
        logs_text, n_sample, label = window_event_sample(w, self.max_logs, focus_ns=culprit)
        primary, _others = rca_focus(w, culprit)
        initial_context = (
            f"Anomaly Score: {scored_window.score:.3f}\n"
            f"{rca_namespaces_line(w, culprit)}\n"
            f"Window: t={w.start_time:.0f}s – t={w.end_time:.0f}s\n"
            f"Total events: {w.log_count} | Distinct templates: {w.template_count}\n"
            f"{label} ({n_sample} lines):\n{logs_text}"
        )

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": initial_context},
        ]

        trace: list[TraceStep] = []
        root_cause = "No se pudo determinar la causa raíz."
        kubectl_cmd = "kubectl get events --all-namespaces --sort-by='.lastTimestamp'"
        confidence = "low"
        seen_actions: set[str] = set()

        for step in range(1, self.max_steps + 1):
            response_text = self._call_llm(messages)
            thought, action, is_final, rc, kc, conf = _parse_response(response_text)

            if is_final:
                trace.append(TraceStep(step=step, thought=thought, action=None, observation=None, is_final=True))
                root_cause = rc or thought
                kubectl_cmd = kc or kubectl_cmd
                confidence = conf
                break

            if action and action not in seen_actions:
                seen_actions.add(action)
                observation = self._run_tool(action)
                trace.append(TraceStep(step=step, thought=thought, action=action, observation=observation))
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"OBSERVATION:\n{observation}\n\nContinue your investigation or provide FINAL answer.",
                })
            else:
                # Sin acción nueva — el modelo usó el formato antiguo (ROOT CAUSE/KUBECTL) o no siguió ReAct
                # Acepta rc/kc si el parser los encontró, aunque no haya FINAL:
                trace.append(TraceStep(step=step, thought=thought, action=None, observation=None, is_final=True))
                root_cause = rc or thought or root_cause
                kubectl_cmd = kc or kubectl_cmd
                confidence = conf if conf != "low" else confidence
                break
        else:
            # Máximo de pasos alcanzado
            messages.append({"role": "user", "content": "Investigation limit reached. Provide your FINAL answer now."})
            final_text = self._call_llm(messages)
            _, _, _, rc, kc, conf = _parse_response(final_text)
            if rc:
                root_cause, kubectl_cmd, confidence = rc, kc or kubectl_cmd, conf

        # Anti-deriva: fallback determinista desde la plantilla de error dominante.
        root_cause = ensure_meaningful_root_cause(root_cause, w)

        return DiagnosisResult(
            window_index=w.index,
            anomaly_score=scored_window.score,
            namespaces={primary} if primary else set(w.focus_namespaces),
            root_cause=root_cause,
            kubectl_command=kubectl_cmd,
            model_version=scored_window.model_version,
            confidence=confidence,
            steps_taken=len(trace),
            react_trace=trace,
            mode="react",
            prompt_user=initial_context,
        )

    def _call_llm(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 500},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _run_tool(self, action: str) -> str:
        if self.dry_run:
            return f"[dry-run] Ejecutaría: {action}"
        result = kubectl_execute(action)
        if result.error and not result.stdout:
            return f"Error: {result.error}"
        return result.stdout or f"Comando sin output (exit {result.returncode})"

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as c:
                resp = c.get(f"{self.host}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                return any(self.model in m for m in models)
        except Exception:
            return False


def _parse_response(text: str) -> tuple[str, str | None, bool, str, str, str]:
    """Devuelve (thought, action, is_final, root_cause, kubectl_cmd, confidence)."""
    thought = ""
    action = None
    is_final = False
    root_cause = ""
    kubectl_cmd = "kubectl get events --all-namespaces --sort-by='.lastTimestamp'"
    confidence = "low"

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("THOUGHT:"):
            thought = line.removeprefix("THOUGHT:").strip()
        elif line.startswith("ACTION:"):
            action = line.removeprefix("ACTION:").strip()
        elif line == "FINAL:":
            is_final = True
        elif line.startswith("ROOT CAUSE:"):
            root_cause = line.removeprefix("ROOT CAUSE:").strip()
        elif line.startswith("KUBECTL:"):
            kubectl_cmd = line.removeprefix("KUBECTL:").strip()
        elif line.startswith("CONFIDENCE:"):
            confidence = line.removeprefix("CONFIDENCE:").strip().lower()

    return thought, action, is_final, root_cause, kubectl_cmd, confidence
