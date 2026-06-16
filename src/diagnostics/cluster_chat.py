"""
Agente conversacional ReAct con acceso de SOLO LECTURA al cluster.

El operador hace una pregunta en lenguaje natural ("¿qué pasa en producción?")
y el agente investiga en vivo. El diseño está endurecido para un modelo base
pequeño (1.5B), que por sí solo divaga y no profundiza:

  1. Triage determinista: el harness SIEMPRE ejecuta `kubectl get pods -A` como
     primer paso (no depende de que el modelo lo pida) y extrae, también de forma
     determinista, los pods con problemas. Esto elimina la divagación y garantiza
     evidencia de alta señal desde el primer instante.
  2. Profundización guiada: con la lista de problemas delante, el modelo elige un
     pod concreto y hace describe/logs. Si no actúa, el harness auto-profundiza en
     el peor pod.
  3. Síntesis con evidencia real: el experto fine-tuneado concluye a partir del
     RESUMEN de problemas (no de un volcado truncado), de modo que ve los fallos.

Seguridad: toda acción pasa por kubectl_toolbox.execute(), que solo permite
describe/get/logs/top. Es imposible que el chat ejecute un comando destructivo.

chat_iter() es un generador que emite eventos para streaming en vivo (SSE).
"""

import re
from collections.abc import Iterator

import httpx

from src.diagnostics.kubectl_toolbox import execute as kubectl_execute

_PLACEHOLDER = re.compile(r"<[^>]+>")

# Estados considerados sanos. Completed/Succeeded son jobs terminados (no fallos).
_HEALTHY_STATUSES = {"Running", "Completed", "Succeeded"}
_TRIAGE_CMD = "kubectl get pods -A"
_MAX_PROBLEMS_SHOWN = 25

# Prioridad de severidad para auto-profundizar en el peor pod primero.
_SEVERITY = {
    "CrashLoopBackOff": 100,
    "OOMKilled": 95,
    "Error": 90,
    "ImagePullBackOff": 85,
    "ErrImagePull": 84,
    "CreateContainerConfigError": 80,
    "ContainerStatusUnknown": 70,
    "Pending": 60,
}

_SYSTEM_PROMPT = """\
You are a Kubernetes SRE assistant with READ-ONLY access to a live cluster.
An automatic triage has ALREADY run `kubectl get pods -A` and given you, as the
first OBSERVATION, a summary that lists the pods WITH PROBLEMS.

Your job: investigate the concrete cause of a failing pod, then answer.

Each turn output EXACTLY one of these two formats:

Format A — investigate ONE specific resource:
THOUGHT: which problem pod you check and why
ACTION: kubectl describe pod REAL-POD-NAME -n REAL-NAMESPACE

Format B — answer:
THOUGHT: summary of the root cause
ANSWER: clear, concise answer in Spanish with the root cause and the fix

Example:
OBSERVATION: 237 pods, 7 con problemas. Pods con problemas:
- llm-app/oauth2-proxy-vllm-xxx  READY=0/1  STATUS=CrashLoopBackOff  RESTARTS=1620
THOUGHT: El proxy vllm está en CrashLoopBackOff; leo sus logs.
ACTION: kubectl logs oauth2-proxy-vllm-xxx -n llm-app --tail=50

CRITICAL rules:
- Use ONLY real pod and namespace names that appear in an OBSERVATION. NEVER write
  placeholders in angle brackets like <pod> or <namespace>.
- Prefer describe or logs of a PROBLEM pod from the triage summary.
- Read-only kubectl only: get, describe, logs, top. Never delete/apply/patch/scale.
- After 1-2 investigations of the culprit, give your ANSWER.
- Output ONE THOUGHT and ONE ACTION (or ANSWER) per turn, nothing else."""

# El experto fine-tuneado sintetiza la conclusión a partir de la evidencia.
_SYNTH_SYSTEM = """\
You are an expert Site Reliability Engineer specialized in Kubernetes.
You receive an operator's question and the evidence collected from the live
cluster by a read-only investigation, including a triage summary of pods with
problems. Produce a clear, concrete answer in Spanish.

Rules:
- Base your answer ONLY on the evidence provided. Do NOT invent pods, images,
  registries, statuses or error messages that are not literally in the evidence.
- If the triage summary lists pods with problems, name the most critical one
  (e.g. CrashLoopBackOff / Error) as the root cause and explain it using the
  describe/logs evidence of that pod.
- You MAY suggest a fix, but present it as a recommendation for the operator.
  Never claim a cause you cannot see in the evidence. Do not invent kubectl
  commands that modify the cluster as if they were verified solutions.
- If the triage summary says there are NO pods with problems, state that the
  cluster appears healthy — do not fabricate a failure."""


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
        self.model = model              # investigador (base, profundiza)
        self.expert_model = expert_model  # experto fine-tuneado (sintetiza)
        self.timeout = timeout
        self.max_steps = max_steps
        self.dry_run = dry_run  # True = no ejecuta kubectl real (para demo/test)

    def chat_iter(self, question: str) -> Iterator[dict]:
        """Generador ReAct híbrido: triage determinista + el base profundiza + el experto concluye.

        Eventos: {"type": "thought"|"action"|"observation"|"answer"|"error", ...}
        """
        transcript: list[tuple[str, str, str]] = []  # (thought, action, observation)
        seen_actions: set[str] = set()
        step = 0

        # ── Paso 1: TRIAGE determinista (no depende del modelo) ───────────────
        step += 1
        yield {"type": "thought", "step": step, "text": "Reviso el estado de todos los pods del cluster…"}
        yield {"type": "action", "step": step, "command": _TRIAGE_CMD}
        pods_output = self._run_tool(_TRIAGE_CMD)
        seen_actions.add(_TRIAGE_CMD)
        digest = _cluster_digest(pods_output)
        yield {"type": "observation", "step": step, "text": digest}
        transcript.append(("Estado general de los pods", _TRIAGE_CMD, pods_output))
        problems = extract_problem_pods(pods_output)

        # ── Profundización DETERMINISTA del pod más severo ────────────────────
        # No dependemos del modelo débil para la evidencia crítica: el harness
        # ejecuta describe+logs del peor pod, garantizando causa raíz real.
        if problems:
            top = _top_problem(problems)
            yield {"type": "thought", "step": step,
                   "text": f"Investigo el pod más crítico: {top['ns']}/{top['name']} ({top['status']})."}
            for cmd in _drill_cmds(top):
                if cmd in seen_actions:
                    continue
                step += 1
                seen_actions.add(cmd)
                yield {"type": "action", "step": step, "command": cmd}
                obs = self._run_tool(cmd)
                yield {"type": "observation", "step": step, "text": obs}
                transcript.append((f"Detalle de {top['name']}", cmd, obs))

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"THOUGHT: Reviso el estado de los pods.\nACTION: {_TRIAGE_CMD}"},
            {"role": "user", "content": (
                f"OBSERVATION:\n{digest}\n\n"
                + ("Ya he investigado el pod más crítico. " if problems else "")
                + "Si necesitas más detalle, investiga OTRO pod problemático con un "
                  "comando exacto, por ejemplo:  kubectl logs NOMBRE -n NAMESPACE --tail=40  "
                  "(usa el nombre del pod SIN el namespace delante). "
                  "Si ya tienes la causa, responde con ANSWER."
            )},
        ]

        # ── Pasos siguientes: profundización opcional guiada por el modelo ────
        for _ in range(2, self.max_steps + 1):
            try:
                response = self._call(messages)
            except Exception as e:
                yield {"type": "error", "text": f"Error consultando el modelo: {e}"}
                yield from self._final_answer(question, transcript)
                return

            thought, action, answer = _parse(response)
            step += 1
            if thought:
                yield {"type": "thought", "step": step, "text": thought}

            if answer or not action or action in seen_actions:
                yield from self._final_answer(question, transcript)
                return

            # Guard: el modelo a veces copia placeholders del prompt (<namespace>…).
            if _PLACEHOLDER.search(action):
                yield {"type": "action", "step": step, "command": action}
                corr = ("Ese comando tiene un placeholder entre < >. No es válido. "
                        "Usa un nombre real de pod/namespace de la lista de problemas, "
                        "por ejemplo haz describe de uno de ellos.")
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
            messages.append({"role": "user", "content": (
                f"OBSERVATION:\n{observation[:1500]}\n\nContinúa o responde con ANSWER."
            )})

        # Agotados los pasos → el experto sintetiza con la evidencia acumulada.
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
        parts = []
        for _thought, action, observation in transcript:
            if _is_broad_pod_listing(action):
                # Sustituir el volcado gigante por el resumen de problemas (alta señal).
                parts.append(f"$ {action}\n{_cluster_digest(observation)}")
            else:
                parts.append(f"$ {action}\n{observation[:1500]}")
        evidence = "\n\n".join(parts) or "(sin comandos ejecutados)"

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


# ──────────────────────────────────────────────────────────────────────────
# Análisis determinista del estado de los pods
# ──────────────────────────────────────────────────────────────────────────

def _is_broad_pod_listing(action: str) -> bool:
    a = action.lower()
    return "get pods" in a and ("-a" in a.split() or "--all-namespaces" in a)


def _int(token: str) -> int:
    m = re.match(r"\d+", token or "")
    return int(m.group()) if m else 0


def _parse_pod_rows(pods_output: str) -> list[dict]:
    """Parsea la salida de `kubectl get pods` (con o sin columna NAMESPACE)."""
    rows: list[dict] = []
    for line in pods_output.splitlines():
        s = line.strip()
        if not s or s.startswith(("NAMESPACE", "NAME ", "...")):
            continue
        cols = s.split()
        # Forma -A: ns name ready status restarts age (ready tiene '/')
        if len(cols) >= 6 and "/" in cols[2]:
            ns, name, ready, status, restarts = cols[0], cols[1], cols[2], cols[3], cols[4]
        # Forma -n <ns>: name ready status restarts age
        elif len(cols) >= 5 and "/" in cols[1]:
            ns, name, ready, status, restarts = "", cols[0], cols[1], cols[2], cols[3]
        else:
            continue
        rows.append({"ns": ns, "name": name, "ready": ready,
                     "status": status, "restarts": _int(restarts)})
    return rows


def _is_problem(row: dict) -> bool:
    status = row["status"]
    if status not in _HEALTHY_STATUSES:
        return True
    # Un pod Running pero no totalmente listo (p.ej. 0/2) es un problema.
    # Los reinicios antiguos en pods Running+ready NO se marcan: un crash activo
    # aparece como CrashLoopBackOff/Error en STATUS, evitando falsos positivos.
    if status == "Running" and "/" in row["ready"]:
        have, _, total = row["ready"].partition("/")
        if _int(have) < _int(total):
            return True
    return False


def extract_problem_pods(pods_output: str) -> list[dict]:
    """Devuelve solo las filas de pods que representan un problema real."""
    return [r for r in _parse_pod_rows(pods_output) if _is_problem(r)]


def _top_problem(problems: list[dict]) -> dict:
    """El pod problemático más severo (para auto-profundizar)."""
    return max(problems, key=lambda p: (_SEVERITY.get(p["status"], 50), p["restarts"]))


def _describe_cmd(pod: dict) -> str:
    if pod["ns"]:
        return f"kubectl describe pod {pod['name']} -n {pod['ns']}"
    return f"kubectl describe pod {pod['name']}"


def _drill_cmds(pod: dict) -> list[str]:
    """Comandos read-only para investigar a fondo un pod (describe + logs)."""
    ns = f" -n {pod['ns']}" if pod["ns"] else ""
    return [
        f"kubectl describe pod {pod['name']}{ns}",
        f"kubectl logs {pod['name']}{ns} --tail=40",
    ]


def _cluster_digest(pods_output: str) -> str:
    """Resumen de alta señal: totales + lista de pods con problemas."""
    rows = _parse_pod_rows(pods_output)
    if not rows:
        # No es una lista de pods (p.ej. salida de describe/logs); devolver tal cual recortado.
        return pods_output[:1500]
    problems = [r for r in rows if _is_problem(r)]
    healthy = len(rows) - len(problems)
    head = f"Resumen del cluster: {len(rows)} pods, {healthy} sanos, {len(problems)} con problemas."
    if not problems:
        return head + "\nNo hay pods con problemas; el cluster parece sano."
    lines = [head, "Pods con problemas:"]
    for p in problems[:_MAX_PROBLEMS_SHOWN]:
        loc = f"{p['ns']}/{p['name']}" if p["ns"] else p["name"]
        lines.append(f"- {loc}  READY={p['ready']}  STATUS={p['status']}  RESTARTS={p['restarts']}")
    if len(problems) > _MAX_PROBLEMS_SHOWN:
        lines.append(f"... y {len(problems) - _MAX_PROBLEMS_SHOWN} más")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# Parsing de la respuesta del modelo
# ──────────────────────────────────────────────────────────────────────────

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
