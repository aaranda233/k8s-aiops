"""
Agente conversacional ReAct con acceso de SOLO LECTURA al cluster.

El operador hace una pregunta en lenguaje natural ("¿qué pasa en producción?")
y el agente investiga en vivo: razona (THOUGHT), ejecuta kubectl de solo lectura
(ACTION, vía kubectl_toolbox que rechaza cualquier escritura), observa el
resultado (OBSERVATION) y repite hasta dar una respuesta (ANSWER).

Seguridad: toda acción pasa por kubectl_toolbox.execute(), que solo permite
describe/get/logs/top. Es imposible que el chat ejecute un comando destructivo.

chat_iter() es un generador que emite eventos para streaming en vivo (SSE).
"""

import re
from collections.abc import Iterator

import httpx

from src.diagnostics.kubectl_toolbox import execute as kubectl_execute

_PLACEHOLDER = re.compile(r"<[^>]+>")

_SYSTEM_PROMPT = """\
You are a Kubernetes SRE assistant with READ-ONLY access to a live cluster.
Answer the user's question by investigating step by step.

Each turn, output EXACTLY one of these two formats:

Format A — investigate:
THOUGHT: what you want to check and why
ACTION: kubectl get pods -A

Format B — answer (when you have enough evidence, or after a few steps):
THOUGHT: summary of findings
ANSWER: clear, concise answer to the user's question in Spanish

CRITICAL rules:
- The ACTION must be a REAL, ready-to-run command. NEVER write placeholders in
  angle brackets like <namespace>, <pod> or <resource>. Use literal real names.
- You do NOT know the namespaces or names in advance. ALWAYS discover them first:
    step 1: kubectl get namespaces        (to see real namespace names)
    step 2: kubectl get pods -A           (or -n <real-ns> once you know it)
    step 3: kubectl describe / logs of a specific real pod
- If the user mentions a place like "producción" that is not a literal namespace,
  first list namespaces and pick the relevant real ones (e.g. default, llm-app...).
- Only read-only kubectl: get, describe, logs, top. Never delete/apply/patch.
- Use ONLY exact names seen in a previous OBSERVATION or in the user's question.
- Be efficient: stop and ANSWER once you have the cause.
- Output ONLY one THOUGHT and one ACTION (or ANSWER) per turn, nothing else."""

# El experto fine-tuneado sintetiza la conclusión a partir de la evidencia
_SYNTH_SYSTEM = """\
You are an expert Site Reliability Engineer specialized in Kubernetes.
You receive an operator's question and the evidence collected from the live
cluster by a read-only investigation. Produce a clear, concrete answer in
Spanish. If a failure is present in the evidence, state the specific root
cause and the concrete fix. Be direct — do not invent data not in the evidence."""


class ClusterChatAgent:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:1.5b",
        expert_model: str = "k8s-rca-orpo:latest",
        timeout: float = 60.0,
        max_steps: int = 5,
        dry_run: bool = False,
    ):
        self.host = host.rstrip("/")
        self.model = model              # investigador (base, sigue bien el ReAct)
        self.expert_model = expert_model  # experto fine-tuneado (sintetiza la conclusión)
        self.timeout = timeout
        self.max_steps = max_steps
        self.dry_run = dry_run  # True = no ejecuta kubectl real (para demo/test)

    def chat_iter(self, question: str) -> Iterator[dict]:
        """Generador ReAct híbrido: el base investiga, el experto concluye.

        Eventos: {"type": "thought"|"action"|"observation"|"answer"|"error", ...}
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        seen_actions: set[str] = set()
        transcript: list[tuple[str, str, str]] = []  # (thought, action, observation)

        for step in range(1, self.max_steps + 1):
            try:
                response = self._call(messages)
            except Exception as e:
                yield {"type": "error", "text": f"Error consultando el modelo: {e}"}
                return

            thought, action, answer = _parse(response)

            if thought:
                yield {"type": "thought", "step": step, "text": thought}

            if answer or not action or action in seen_actions:
                # Invariante: hay que investigar al menos una vez antes de concluir.
                # Si el base intenta responder sin evidencia, lo empujamos a investigar.
                if not transcript and step < self.max_steps:
                    messages.append({
                        "role": "user",
                        "content": "Primero investiga con al menos un comando kubectl read-only "
                                   "(get/describe/logs) antes de responder.",
                    })
                    continue
                # El EXPERTO sintetiza la conclusión a partir de la evidencia.
                yield from self._final_answer(question, transcript)
                return

            # Guard: el modelo a veces copia placeholders del prompt (<namespace>...).
            # No ejecutamos eso; le devolvemos una corrección para que use nombres reales.
            if _PLACEHOLDER.search(action):
                yield {"type": "action", "step": step, "command": action}
                corr = ("Ese comando tiene un placeholder entre < >. No es válido. "
                        "Descubre nombres reales primero: ejecuta  kubectl get namespaces  "
                        "y luego  kubectl get pods -A  — sin placeholders.")
                yield {"type": "observation", "step": step, "text": corr}
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"OBSERVATION:\n{corr}"})
                continue

            seen_actions.add(action)
            yield {"type": "action", "step": step, "command": action}

            observation = self._run_tool(action)
            yield {"type": "observation", "step": step, "text": observation}
            transcript.append((thought, action, observation))

            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"OBSERVATION:\n{observation}\n\nContinue or give ANSWER.",
            })

        # Agotados los pasos → el experto sintetiza con la evidencia acumulada
        yield from self._final_answer(question, transcript)

    def _final_answer(self, question: str, transcript: list[tuple[str, str, str]]) -> Iterator[dict]:
        """El modelo experto (fine-tuneado) concluye a partir de la evidencia."""
        yield {"type": "thought", "text": "Sintetizando diagnóstico…"}
        try:
            answer = self._synthesize(question, transcript)
        except Exception as e:
            yield {"type": "error", "text": f"Error en la síntesis: {e}"}
            return
        yield {"type": "answer", "text": answer}

    def _synthesize(self, question: str, transcript: list[tuple[str, str, str]]) -> str:
        evidence = "\n\n".join(
            f"$ {action}\n{observation[:600]}"
            for _thought, action, observation in transcript
        ) or "(sin comandos ejecutados)"

        messages = [
            {"role": "system", "content": _SYNTH_SYSTEM},
            {"role": "user", "content": (
                f"Pregunta del operador: {question}\n\n"
                f"Evidencia recogida del cluster (kubectl read-only):\n{evidence}\n\n"
                f"Responde a la pregunta de forma clara en español. Si hay un fallo, "
                f"explica la causa raíz concreta y cómo solucionarlo."
            )},
        ]
        return self._call(messages, model=self.expert_model)

    def _call(self, messages: list[dict], model: str | None = None) -> str:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 400},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    def chat(self, question: str) -> dict:
        """Versión no-streaming: recopila todos los eventos y devuelve el resultado."""
        steps, answer = [], ""
        for ev in self.chat_iter(question):
            if ev["type"] in ("answer", "error"):
                answer = ev["text"]
            else:
                steps.append(ev)
        return {"steps": steps, "answer": answer}

    def _run_tool(self, action: str) -> str:
        if self.dry_run:
            return f"[dry-run] Ejecutaría: {action}"
        result = kubectl_execute(action)  # solo lectura garantizada
        if result.error and not result.stdout:
            return f"Error: {result.error}"
        return result.stdout or f"(sin salida, exit {result.returncode})"

    def health_check(self) -> bool:
        try:
            with httpx.Client(timeout=5.0) as c:
                resp = c.get(f"{self.host}/api/tags")
                models = [m["name"] for m in resp.json().get("models", [])]
                return any(self.model.split(":")[0] in m for m in models)
        except Exception:
            return False


def _parse(text: str) -> tuple[str, str | None, str | None]:
    """Devuelve (thought, action, answer). action/answer son None si no aparecen.

    Tolerante con el modelo de 1.5B: si no emite la línea 'ACTION:' con el
    formato exacto, extrae igualmente cualquier comando 'kubectl ...' del texto.
    ANSWER puede ser multilínea: se captura todo lo que sigue a 'ANSWER:'.
    """
    thought = ""
    action = None
    answer = None

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("THOUGHT:"):
            thought = stripped.removeprefix("THOUGHT:").strip()
        elif stripped.startswith("ACTION:"):
            action = _clean_cmd(stripped.removeprefix("ACTION:").strip())
        elif stripped.startswith("ANSWER:"):
            first = stripped.removeprefix("ANSWER:").strip()
            rest = lines[idx + 1:]
            answer = "\n".join([first, *rest]).strip()
            break

    # Fallback tolerante: ni ACTION explícita ni ANSWER → buscar un 'kubectl ...'
    # en cualquier parte del texto (el modelo a veces lo escribe en prosa o en ```).
    if action is None and answer is None:
        for line in lines:
            cleaned = _clean_cmd(line)
            if cleaned.startswith("kubectl "):
                action = cleaned
                break

    return thought, action, answer


def _clean_cmd(s: str) -> str:
    """Limpia un comando: quita backticks, viñetas y prefijos de prosa."""
    s = s.strip().strip("`").strip()
    for prefix in ("- ", "* ", "$ ", "ACTION:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    # Si hay texto antes de 'kubectl', recortar desde ahí
    idx = s.find("kubectl ")
    if idx > 0 and idx <= 40:  # 'kubectl' aparece tras algo de prosa corta
        s = s[idx:]
    return s.strip()
