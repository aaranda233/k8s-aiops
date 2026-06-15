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

from collections.abc import Iterator

import httpx

from src.diagnostics.kubectl_toolbox import execute as kubectl_execute

_SYSTEM_PROMPT = """\
You are a Kubernetes SRE assistant with READ-ONLY access to a live cluster.
Answer the user's question by investigating step by step.

Each turn, output EXACTLY one of these two formats:

Format A — investigate:
THOUGHT: <what you want to check and why>
ACTION: kubectl <read-only command: describe | get | logs | top>

Format B — answer (when you have enough evidence, or after a few steps):
THOUGHT: <summary of findings>
ANSWER: <clear, concise answer to the user's question in Spanish>

Rules:
- Only read-only kubectl (describe, get, logs, top). Never delete/apply/patch.
- Use exact resource names and namespaces seen in previous observations.
- Be efficient: stop and ANSWER as soon as you can answer the question.
- Output ONLY the format above, nothing else."""


class ClusterChatAgent:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen2.5:1.5b",
        timeout: float = 60.0,
        max_steps: int = 5,
        dry_run: bool = False,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_steps = max_steps
        self.dry_run = dry_run  # True = no ejecuta kubectl real (para demo/test)

    def chat_iter(self, question: str) -> Iterator[dict]:
        """Generador que emite eventos del ciclo ReAct para streaming.

        Eventos: {"type": "thought"|"action"|"observation"|"answer"|"error", ...}
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        seen_actions: set[str] = set()

        for step in range(1, self.max_steps + 1):
            try:
                response = self._call(messages)
            except Exception as e:
                yield {"type": "error", "text": f"Error consultando el modelo: {e}"}
                return

            thought, action, answer = _parse(response)

            if thought:
                yield {"type": "thought", "step": step, "text": thought}

            if answer:
                yield {"type": "answer", "text": answer}
                return

            if not action or action in seen_actions:
                # Sin acción nueva → forzar respuesta final
                yield {"type": "answer", "text": thought or "No pude completar la investigación."}
                return

            seen_actions.add(action)
            yield {"type": "action", "step": step, "command": action}

            observation = self._run_tool(action)
            yield {"type": "observation", "step": step, "text": observation}

            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"OBSERVATION:\n{observation}\n\nContinue or give ANSWER.",
            })

        # Agotados los pasos → pedir respuesta final
        messages.append({"role": "user", "content": "Investigation limit reached. Give your ANSWER now."})
        try:
            final = self._call(messages)
            _, _, answer = _parse(final)
            yield {"type": "answer", "text": answer or "No pude llegar a una conclusión con la información disponible."}
        except Exception as e:
            yield {"type": "error", "text": f"Error en la respuesta final: {e}"}

    def chat(self, question: str) -> dict:
        """Versión no-streaming: recopila todos los eventos y devuelve el resultado."""
        steps, answer = [], ""
        for ev in self.chat_iter(question):
            if ev["type"] == "answer":
                answer = ev["text"]
            elif ev["type"] == "error":
                answer = ev["text"]
            else:
                steps.append(ev)
        return {"steps": steps, "answer": answer}

    def _call(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 400},
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

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
            action = stripped.removeprefix("ACTION:").strip()
        elif stripped.startswith("ANSWER:"):
            # Capturar el resto del texto (puede ser multilínea)
            first = stripped.removeprefix("ANSWER:").strip()
            rest = lines[idx + 1:]
            answer = "\n".join([first, *rest]).strip()
            break

    return thought, action, answer
