"""
Agente híbrido de dos fases para Root Cause Analysis.

Fase 1 — Investigador (modelo base, qwen2.5:1.5b vanilla):
  Lee los eventos, planea qué comandos kubectl ejecutaría, razona paso a paso.
  Modelo pequeño sin fine-tune → sigue instrucciones nuevas sin problemas.

Fase 2 — Experto (modelo fine-tuneado, k8s-rca-orpo):
  Recibe los eventos originales + el plan de investigación acumulado.
  Produce el diagnóstico final en el formato ROOT CAUSE / KUBECTL que conoce.

Resultado: mejor razonamiento (fase 1) + mejor formato/dominio K8s (fase 2).
"""

from dataclasses import dataclass

import httpx

from src.diagnostics.command_builder import build_command, build_remediation
from src.diagnostics.kubectl_toolbox import execute as kubectl_execute
from src.diagnostics.ollama_rca import (
    DiagnosisResult,
    ensure_meaningful_root_cause,
    parse_diagnosis,
    rca_focus,
    rca_namespaces_line,
    window_event_sample,
)

# GBNF grammar que fuerza ROOT CAUSE: ... \n KUBECTL: kubectl ... a nivel de token.
# Elimina el fallo de formato independientemente del contexto extra que recibe el experto.
_GRAMMAR_GBNF = r"""root         ::= "ROOT CAUSE: " rc-text "\nKUBECTL: " kubectl-text
rc-text      ::= [^\n]+ (" " [^\n]+)*
kubectl-text ::= "kubectl " [^\n]+
"""

_INVESTIGATOR_SYSTEM = """\
You are a Kubernetes SRE assistant. Given anomalous cluster events, plan your investigation.

For each step output exactly:
THOUGHT: <razonamiento en español, breve>
ACTION: kubectl <read-only command, UN solo namespace>

When you have enough to diagnose (or after 3 steps), output:
THOUGHT: <resumen final en español>
DONE

Rules:
- THOUGHT siempre en español, una frase.
- ACTION: comando read-only (describe, get, logs, top) con UN solo -n <namespace>.
  Nunca pongas varios namespaces separados por comas."""

_EXPERT_SYSTEM = """\
You are an expert Site Reliability Engineer (SRE) specialized in Kubernetes.
You receive raw Kubernetes events from a time window flagged as anomalous by an ML model.
Your task:
1. Identify the root cause in 2-3 sentences.
2. Propose ONE specific kubectl command to investigate or mitigate.

IMPORTANT:
- Respond ALWAYS in Spanish. No preamble or extra text before ROOT CAUSE.
- ROOT CAUSE: máximo 3 frases. PROHIBIDO listas, viñetas, pasos numerados,
  markdown o bloques de código.
- KUBECTL: UN solo comando en UNA línea, con UN solo namespace (-n <uno>).

Output format (strict):
ROOT CAUSE: <explicación breve en español, máx 3 frases>
KUBECTL: <un comando exacto>"""


@dataclass
class InvestigationStep:
    step: int
    thought: str
    action: str | None
    observation: str | None
    is_done: bool = False


@dataclass
class HybridReActAgent:
    host: str = "http://localhost:11434"
    base_model: str = "qwen2.5:1.5b"
    expert_model: str = "k8s-rca-orpo:latest"
    max_logs: int = 40
    timeout: float = 120.0
    max_steps: int = 3
    dry_run: bool = True
    retriever: object = None   # IncidentRetriever opcional (RAG): casos pasados

    def diagnose(self, scored_window) -> DiagnosisResult:
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

        # RAG: el contexto recuperado solo se inyecta al INVESTIGADOR (modelo base,
        # que se beneficia del contexto y no es sensible al formato). NO se inyecta
        # al EXPERTO fine-tuneado: el bloque RAG le hace abandonar el formato
        # ROOT CAUSE/KUBECTL y divagar en modo tutorial (hallazgo verificado).
        investigator_context = initial_context
        if self.retriever is not None:
            from src.diagnostics.incident_retriever import rag_context
            ctx = rag_context(self.retriever.retrieve(logs_text, k=2))
            if ctx:
                investigator_context = f"{ctx}\n\n{initial_context}"

        # Fase 1: investigador (base model) — con RAG si lo hay
        investigation_steps = self._investigate(investigator_context)

        # Fase 2: experto (fine-tuned) — contexto LIMPIO (eventos) + notas, sin RAG
        root_cause, kubectl_cmd = self._expert_diagnose(initial_context, investigation_steps)
        # Anti-deriva: si el experto divaga/se disculpa, fallback determinista
        # derivado de la plantilla de error dominante (nunca "sin causa" si hay errores).
        root_cause = ensure_meaningful_root_cause(root_cause, w)
        # Comando dirigido + remediación reversible (deterministas, namespace correcto).
        kubectl_cmd = build_command(logs_text, primary, root_cause, kubectl_cmd)
        remediation = build_remediation(logs_text, primary, root_cause)

        return DiagnosisResult(
            window_index=w.index,
            anomaly_score=scored_window.score,
            namespaces={primary} if primary else set(w.focus_namespaces),
            root_cause=root_cause,
            kubectl_command=kubectl_cmd,
            model_version=scored_window.model_version,
            confidence="medium",
            steps_taken=len(investigation_steps),
            react_trace=investigation_steps,
            mode="hybrid",
            prompt_user=initial_context,
            remediation_command=remediation,
        )

    def _investigate(self, initial_context: str) -> list[InvestigationStep]:
        messages = [
            {"role": "system", "content": _INVESTIGATOR_SYSTEM},
            {"role": "user", "content": initial_context},
        ]
        steps: list[InvestigationStep] = []
        seen_actions: set[str] = set()

        for i in range(1, self.max_steps + 1):
            response = self._call(messages, model=self.base_model, num_predict=250)
            thought, action, is_done = _parse_investigator(response)

            if is_done or not action or action in seen_actions:
                steps.append(InvestigationStep(step=i, thought=thought, action=None,
                                               observation=None, is_done=True))
                break

            seen_actions.add(action)
            observation = self._run_tool(action)
            steps.append(InvestigationStep(step=i, thought=thought, action=action,
                                           observation=observation))
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"OBSERVATION:\n{observation}\n\nContinue or output DONE.",
            })

        return steps

    def _expert_diagnose(self, initial_context: str, steps: list[InvestigationStep]) -> tuple[str, str]:
        plan_lines = []
        for s in steps:
            if s.thought:
                plan_lines.append(f"  Step {s.step} thought: {s.thought}")
            if s.action:
                plan_lines.append(f"  Step {s.step} action: {s.action}")
            if s.observation and not s.observation.startswith("[dry-run]"):
                plan_lines.append(f"  Step {s.step} observation: {s.observation[:200]}")

        plan_section = ""
        if plan_lines:
            plan_section = "\n\n[Investigation notes from first-pass analysis:]\n" + "\n".join(plan_lines)

        user_content = initial_context + plan_section
        # Grammar-constrained sampling via /api/generate — garantiza formato ROOT CAUSE/KUBECTL
        # independientemente del tamaño del contexto.
        root_cause, kubectl_cmd = self._call_expert_with_grammar(user_content)
        return root_cause, kubectl_cmd

    def _call_expert_with_grammar(self, user_content: str) -> tuple[str, str]:
        """Llama al experto con GBNF grammar para formato garantizado."""
        prompt = (
            f"<|im_start|>system\n{_EXPERT_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        payload = {
            "model": self.expert_model,
            "prompt": prompt,
            "stream": False,
            "grammar": _GRAMMAR_GBNF,
            # num_predict bajo + stop: cortan la divagación tipo tutorial (listas,
            # markdown, pasos numerados) si la grammar no la aplica el runtime.
            "stop": ["\n\n", "\n1.", "\n- ", "\n#", "```", "\nROOT CAUSE", "\nAnalysis", "\nPaso"],
            "options": {"temperature": 0.1, "num_predict": 160, "num_ctx": 2048},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/generate", json=payload)
            resp.raise_for_status()
        text = resp.json()["response"].strip()
        return parse_diagnosis(text)

    def _call(self, messages: list[dict], model: str, num_predict: int = 300) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": num_predict},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def _run_tool(self, action: str) -> str:
        if self.dry_run:
            return f"[dry-run] Would execute: {action}"
        result = kubectl_execute(action)
        if result.error and not result.stdout:
            return f"Error: {result.error}"
        return result.stdout or f"Empty output (exit {result.returncode})"

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as c:
                resp = c.get(f"{self.host}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                return (
                    any(self.base_model in m for m in models) and
                    any(self.expert_model.split(":")[0] in m for m in models)
                )
        except Exception:
            return False


def _parse_investigator(text: str) -> tuple[str, str | None, bool]:
    """Devuelve (thought, action, is_done)."""
    thought = ""
    action = None
    is_done = False

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("THOUGHT:"):
            thought = line.removeprefix("THOUGHT:").strip()
        elif line.startswith("ACTION:"):
            action = line.removeprefix("ACTION:").strip()
        elif line == "DONE":
            is_done = True

    return thought, action, is_done
