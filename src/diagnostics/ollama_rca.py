"""
Capa 3 — Root Cause Analysis via SLM local (Ollama).
Solo se activa de forma reactiva cuando score >= threshold.
"""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import httpx

from src.diagnostics.command_builder import build_command, build_remediation, explain_command

if TYPE_CHECKING:
    pass

# Comando kubectl real (verbo conocido) — evita capturar cabeceras como
# "KUBECTL COMMANDS:" que el modelo a veces emite en vez de un comando.
_KUBECTL_RE = re.compile(
    r"(kubectl\s+(?:get|describe|logs|top|rollout|scale|delete|apply|patch|set|"
    r"edit|exec|cordon|drain|annotate|label|create|expose|run|explain)\b.*)",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = """\
You are an expert Site Reliability Engineer (SRE) specialized in Kubernetes.
You receive raw Kubernetes events from a time window flagged as anomalous by an ML model.
Your task:
1. Identify the root cause in 2-3 sentences.
2. Propose ONE specific kubectl command to investigate or mitigate.

IMPORTANT:
- Respond ALWAYS in Spanish. No preamble before ROOT CAUSE.
- ROOT CAUSE: máximo 3 frases. PROHIBIDO listas, viñetas, pasos numerados,
  markdown o bloques de código.
- KUBECTL: UN solo comando en UNA línea, con UN solo namespace (-n <uno>).

Output format (strict, no extra text):
ROOT CAUSE: <explicación breve en español, máx 3 frases>
KUBECTL: <un comando exacto>"""

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


_SAMPLE_ERROR_MIN = 5  # mínimo de logs de error para liderar la muestra con ellos
_MAX_TEMPLATES = 12    # nº máximo de patrones distintos a mostrar al SLM


def cluster_error_templates(records, max_templates: int = _MAX_TEMPLATES,
                            max_chars: int = _MAX_SAMPLE_CHARS) -> tuple[str, int]:
    """Agrupa logs de error por (namespace, plantilla) y los resume para el SLM.

    En vez de volcar 40 líneas casi idénticas (que diluyen la señal y consumen el
    presupuesto de num_ctx), emite una línea por patrón con su frecuencia y un
    ejemplo real, ordenadas de más a menos frecuente:

        12× [postgresql] FATAL: role "<*>" does not exist
             ej: FATAL: role "$(POSTGRES_USER)" does not exist

    Devuelve (texto, nº de patrones distintos).
    """
    groups: dict[tuple, dict] = {}
    for r in records:
        ns = (getattr(r, "namespace", "") or "").strip()
        template = (getattr(r, "template", "") or getattr(r, "raw", "") or "").strip()
        key = (ns, getattr(r, "cluster_id", template))
        g = groups.get(key)
        if g is None:
            groups[key] = {"count": 1, "template": template, "namespace": ns,
                           "example": (getattr(r, "raw", "") or "").strip()}
        else:
            g["count"] += 1
    # Más frecuentes primero (estable: empates conservan orden de aparición).
    ranked = sorted(groups.values(), key=lambda g: g["count"], reverse=True)[:max_templates]
    lines: list[str] = []
    for g in ranked:
        ns = f"[{g['namespace']}] " if g["namespace"] else ""
        template = g["template"]
        if len(template) > _MAX_LINE_CHARS:
            template = template[:_MAX_LINE_CHARS] + "…"
        lines.append(f"  {g['count']}× {ns}{template}")
        example = g["example"]
        if example and example != g["template"]:
            if len(example) > _MAX_LINE_CHARS:
                example = example[:_MAX_LINE_CHARS] + "…"
            lines.append(f"       ej: {example}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars]
    return text, len(groups)


def rca_focus(window, primary_override: str | None = None) -> tuple[str, list[str]]:
    """Devuelve (namespace_primario, otros_con_errores).

    El primario es el culpable: si el detector lo señala (primary_override) se usa
    ese; si no, el namespace dominante por errores. El RCA debe centrarse en UNA
    causa, no diluirse.
    """
    focus = list(getattr(window, "focus_namespaces", None) or [])
    primary = primary_override or getattr(window, "primary_namespace", None) or (focus[0] if focus else "")
    others = [ns for ns in focus if ns != primary]
    return primary, others


def rca_namespaces_line(window, primary_override: str | None = None) -> str:
    """Línea de cabecera para el prompt: lidera con el namespace culpable."""
    primary, others = rca_focus(window, primary_override)
    if not primary:
        return "Namespaces affected: (desconocido)"
    line = f"Namespace afectado: {primary}"
    if others:
        line += f"\nOtros namespaces con errores (contexto): {', '.join(others)}"
    return line


def window_event_sample(window, max_logs: int = 40, focus_ns: str | None = None) -> tuple[str, int, str]:
    """Muestra para el RCA priorizando los logs de error si los hay.

    Cuando la anomalía la dispara la severidad (volumen anormal de logs de error),
    el RCA debe ver ESAS líneas, no logs recientes al azar de toda la ventana.
    Si los errores vienen estructurados (error_records), se agrupan por plantilla
    y se FILTRAN al namespace culpable (focus_ns; si no, el dominante por errores)
    para no diluir la señal. Devuelve (texto, n_líneas, etiqueta).
    """
    error_logs = getattr(window, "error_logs", None) or []
    records = getattr(window, "error_records", None) or []
    if len(error_logs) >= _SAMPLE_ERROR_MIN and records:
        primary = focus_ns or getattr(window, "primary_namespace", None)
        focused = [r for r in records
                   if not primary or (getattr(r, "namespace", "") or "") == primary]
        focused = focused or records  # nunca perder señal si el filtro vacía
        text, distinct = cluster_error_templates(focused)
        ns_tag = f" en {primary}" if primary else ""
        return text, len(focused), f"Error patterns{ns_tag} ({distinct} distinct templates)"
    if len(error_logs) >= _SAMPLE_ERROR_MIN:
        text, n = build_event_sample(error_logs, max_logs)
        return text, n, f"Error log sample ({len(error_logs)} error lines in window)"
    text, n = build_event_sample(window.raw_logs, max_logs)
    return text, n, "Event sample"


def parse_diagnosis(text: str) -> tuple[str, str]:
    """Extrae (root_cause, kubectl) de la salida del modelo de forma tolerante.

    Acepta variaciones de formato (markdown, mayúsculas, espacios). Si no encuentra
    el patrón estricto, usa el texto del modelo como causa (mejor que "no parseable")
    y un kubectl por defecto. Nunca devuelve "Could not parse" si el modelo dijo algo.
    """
    lines = text.splitlines()
    root_cause = None
    for i, raw in enumerate(lines):
        line = raw.strip().lstrip("#*->").strip()  # tolera markdown/viñetas
        if line.lower().startswith("root cause"):
            after = line.split(":", 1)[1].strip() if ":" in line else ""
            if after:
                root_cause = after
            else:
                # Cabecera sin contenido en la misma línea → tomar las siguientes
                # hasta la sección de comandos.
                rest = []
                for nxt in lines[i + 1:]:
                    s = nxt.strip().lstrip("#*->").strip()
                    if not s or s.lower().startswith("kubectl"):
                        break
                    rest.append(s)
                root_cause = " ".join(rest).strip() or None
            break

    # kubectl: primer comando REAL en cualquier parte del texto (ignora cabeceras).
    m = _KUBECTL_RE.search(text)
    kubectl_cmd = m.group(1).strip().strip("`").strip() if m else None

    if not root_cause:
        # Fallback: usar el texto del modelo (sin las líneas de comando) como causa.
        cleaned = [l.strip().lstrip("#*->").strip() for l in lines
                   if l.strip() and "kubectl" not in l.lower()]
        root_cause = " ".join(cleaned) if cleaned else "Sin causa raíz determinable."
    # Seguridad: quitar un prefijo "ROOT CAUSE:" redundante si quedó.
    root_cause = re.sub(r"^\s*root cause\s*:?\s*", "", root_cause, flags=re.IGNORECASE).strip() or root_cause
    # Salvaguarda de CALIDAD: el modelo a veces divaga en modo tutorial (markdown,
    # listas, bloques de código). Limpiar y truncar a 2-3 frases concisas.
    root_cause = _concise(root_cause)
    kubectl_cmd = sanitize_kubectl(kubectl_cmd) if kubectl_cmd else _DEFAULT_KUBECTL
    return root_cause, kubectl_cmd


def _concise(text: str, max_sentences: int = 3, max_chars: int = 320) -> str:
    """Limpia markdown/listas/código y trunca a unas pocas frases (anti-tutorial)."""
    s = re.sub(r"```.*?```", " ", text, flags=re.DOTALL).replace("```", " ")
    s = re.sub(r"[*`#]", "", s)                          # markdown bold/code/headers
    # marcadores de lista en CUALQUIER posición (no solo a inicio de línea, porque
    # el texto puede venir ya unido en una sola línea): "1. " "2) " y viñetas.
    s = re.sub(r"(^|\s)\d+[.)]\s+", " ", s)
    s = re.sub(r"(^|\s)[-•]\s+", " ", s)
    s = re.sub(r"(?i)^\s*(an[aá]lisis|analysis)\b[:\s]*", "", s)  # preámbulo
    s = re.sub(r"\s+", " ", s).strip()
    parts = re.split(r"(?<=[.!?])\s+", s)
    out = " ".join(parts[:max_sentences]).strip()
    return (out[:max_chars].rstrip() or s[:max_chars]).strip()


# Marcadores de "deriva": el SLM se disculpa, pide datos o entra en modo tutorial
# en vez de dar la causa raíz. Detectarlos permite sustituir por un fallback útil.
_DRIFT_MARKERS = (
    "lo siento", "i'm sorry", "i am sorry", "as an ai", "as a language model",
    "parece que falta", "parece que hay un error en", "no se puede identificar",
    "necesito más información", "necesito mas informacion",
    "no puedo ayudar", "no puedo continuar", "no puedo determinar",
    "here are", "additional steps", "first step", "primer paso", "en respuesta a",
    "confusión anterior", "confusion anterior",
)
_DRIFT_STEP_RE = re.compile(r"\b(?:step|paso)\s*\d", re.IGNORECASE)
# Causas "débiles" que no aportan nada (incluye nuestros propios fallbacks).
_WEAK_RC = (
    "sin causa raíz determinable", "sin causa raiz determinable",
    "could not parse", "no se pudo determinar",
)


def _looks_like_drift(text: str) -> bool:
    """True si la salida del modelo es una disculpa/relleno/tutorial, no un diagnóstico."""
    if not text or not text.strip():
        return True
    low = text.lower()
    if any(m in low for m in _DRIFT_MARKERS):
        return True
    return bool(_DRIFT_STEP_RE.search(low))


def _is_weak_root_cause(text: str) -> bool:
    low = (text or "").lower().strip()
    return (not low) or any(w in low for w in _WEAK_RC)


def synthesize_root_cause(window) -> str:
    """Causa raíz determinista a partir de la plantilla de error dominante.

    Cuando el SLM divaga o no concluye, NO mostramos "sin causa": tenemos las
    plantillas Drain3, así que describimos el patrón de error real observado.
    Devuelve "" si la ventana no tiene errores estructurados (nada que sintetizar).
    """
    records = getattr(window, "error_records", None) or []
    primary = getattr(window, "primary_namespace", None)
    focused = [r for r in records
               if not primary or (getattr(r, "namespace", "") or "") == primary] or records
    if not focused:
        return ""
    counts: dict[str, int] = {}
    for r in focused:
        tmpl = (getattr(r, "template", "") or getattr(r, "raw", "") or "").strip()
        if tmpl:
            counts[tmpl] = counts.get(tmpl, 0) + 1
    if not counts:
        return ""
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ns = primary or (list(getattr(window, "focus_namespaces", None) or ["?"]) or ["?"])[0]
    main_t, main_c = top[0]
    if len(main_t) > _MAX_LINE_CHARS:
        main_t = main_t[:_MAX_LINE_CHARS] + "…"
    msg = f"El namespace «{ns}» acumula {main_c} errores recurrentes del tipo: «{main_t}»."
    if len(top) > 1:
        msg += f" También aparece: «{top[1][0]}» ({top[1][1]}×)."
    return msg


def ensure_meaningful_root_cause(root_cause: str, window) -> str:
    """Sustituye un diagnóstico en deriva/vacío por el fallback determinista."""
    if _looks_like_drift(root_cause) or _is_weak_root_cause(root_cause):
        synth = synthesize_root_cause(window)
        if synth:
            return synth
    return root_cause


def sanitize_kubectl(cmd: str) -> str:
    """Convierte el kubectl propuesto en un comando válido y de una sola línea.

    - '-n a, b, c' (multi-namespace, inválido) -> '-n a' (el primario).
    - corta la cola de pipes/redirecciones (| grep | awk, >, etc.).
    - recursos cluster-scoped (node/nodes/pv) no llevan -n.
    - comandos con placeholders <...> (no ejecutables) -> kubectl por defecto.
    """
    cmd = (cmd or "").strip()
    # un solo comando: cortar en el primer pipe/redirección
    cmd = re.split(r"\s*[|>;]", cmd, maxsplit=1)[0].strip()
    cmd = cmd.replace("`", "").strip()  # backticks sueltos (en cualquier posición)
    # -n con lista separada por comas -> primer namespace
    cmd = re.sub(r"(-n\s+)([A-Za-z0-9-]+)\s*,[\sA-Za-z0-9,-]*", r"\1\2", cmd)
    # recursos no-namespaced: quitar -n
    if re.search(r"\b(node|nodes|pv|persistentvolume|persistentvolumes|namespace|namespaces|ns)\b", cmd):
        cmd = re.sub(r"\s+-n\s+\S+", "", cmd)
    cmd = re.sub(r"\s+", " ", cmd).strip()
    # placeholders no ejecutables (<nombre-pod>, <namespace>...) -> comando por defecto
    if not cmd or "<" in cmd or ">" in cmd or not cmd.startswith("kubectl "):
        return _DEFAULT_KUBECTL
    return cmd


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
    prompt_user: str = ""   # input exacto que recibió el modelo (para el dataset de feedback)
    remediation_command: str = ""  # acción reversible propuesta (shadow); "" si manual
    command_explanation: str = ""      # qué hace el comando de investigación
    remediation_explanation: str = ""  # qué hace el comando de remediación


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
        culprit = getattr(scored_window, "culprit_namespace", "") or None
        logs_text, n_sample, label = window_event_sample(w, self.max_logs, focus_ns=culprit)
        primary, _others = rca_focus(w, culprit)

        user_msg = (
            f"Anomaly Score: {scored_window.score:.3f}\n"
            f"{rca_namespaces_line(w, culprit)}\n"
            f"Window: t={w.start_time:.0f}s – t={w.end_time:.0f}s\n"
            f"Total events: {w.log_count} | Distinct templates: {w.template_count}\n"
            f"{label} ({n_sample} lines):\n{logs_text}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "stream": False,
            # stop: corta la divagación tipo tutorial (listas, markdown, pasos).
            "options": {
                "temperature": 0.1,
                "num_predict": 300,
                "stop": ["\n\n", "\n1.", "\n- ", "\n#", "```", "\nPaso", "\nStep"],
            },
        }

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()

        text = resp.json()["message"]["content"].strip()
        root_cause, kubectl_cmd = parse_diagnosis(text)
        root_cause = ensure_meaningful_root_cause(root_cause, w)
        kubectl_cmd = build_command(logs_text, primary, root_cause, kubectl_cmd)
        remediation = build_remediation(logs_text, primary, root_cause)

        return DiagnosisResult(
            window_index=w.index,
            anomaly_score=scored_window.score,
            namespaces={primary} if primary else set(w.focus_namespaces),
            root_cause=root_cause,
            kubectl_command=kubectl_cmd,
            model_version=scored_window.model_version,
            prompt_user=user_msg,
            remediation_command=remediation,
            command_explanation=explain_command(kubectl_cmd),
            remediation_explanation=explain_command(remediation),
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
