"""
Tests de la capa RCA single-shot y utilidades compartidas (src/diagnostics/ollama_rca.py).

Cubre el parseo tolerante (parse_diagnosis), el acotado de la muestra de eventos
(build_event_sample — el fix del bug 'Could not parse root cause') y diagnose()
con la red mockeada.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from src.diagnostics import ollama_rca
from src.diagnostics.ollama_rca import (
    _DEFAULT_KUBECTL,
    OllamaRCA,
    build_event_sample,
    parse_diagnosis,
    sanitize_kubectl,
    window_event_sample,
)

# ── parse_diagnosis: tolerante ──────────────────────────────────────────────

@pytest.mark.unit
def test_parse_strict_format():
    rc, kc = parse_diagnosis("ROOT CAUSE: OOMKilled en api\nKUBECTL: kubectl describe pod api")
    assert rc == "OOMKilled en api"
    assert kc == "kubectl describe pod api"


@pytest.mark.unit
def test_parse_tolerates_markdown_and_case():
    rc, kc = parse_diagnosis("**Root Cause:** disco lleno\n**KUBECTL:** kubectl get pvc")
    assert "disco lleno" in rc
    assert kc == "kubectl get pvc" or kc.startswith("kubectl get")


@pytest.mark.unit
def test_parse_fallback_uses_model_text_not_could_not_parse():
    """Sin formato estricto, usa el texto del modelo (no 'Could not parse')."""
    rc, kc = parse_diagnosis("El pod está en CrashLoopBackOff por un error de OIDC 404.")
    assert "CrashLoopBackOff" in rc
    assert rc != "Could not parse root cause."
    assert kc == _DEFAULT_KUBECTL  # kubectl por defecto cuando no hay comando


@pytest.mark.unit
def test_parse_extracts_bare_kubectl_command():
    rc, kc = parse_diagnosis("Algo va mal\nkubectl logs api -n prod --tail=20")
    assert kc == "kubectl logs api -n prod --tail=20"


@pytest.mark.unit
def test_parse_real_model_format_header_and_numbered_list():
    """Formato real del modelo: 'KUBECTL COMMANDS:' (cabecera) + lista numerada."""
    text = (
        "ROOT CAUSE: The PostgreSQL database is running out of memory and "
        "dropping connections.\n"
        "KUBECTL COMMANDS:\n"
        "1. kubectl describe pod postgres-0 -n db\n"
        "2. kubectl top pod -n db"
    )
    rc, kc = parse_diagnosis(text)
    # El prefijo 'ROOT CAUSE:' se elimina; no se captura la cabecera como comando
    assert rc.startswith("The PostgreSQL")
    assert "ROOT CAUSE" not in rc
    assert kc == "kubectl describe pod postgres-0 -n db"
    assert "COMMANDS" not in kc


@pytest.mark.unit
def test_parse_multiline_root_cause_after_header():
    text = (
        "ROOT CAUSE:\n"
        "The node ran out of disk space.\n"
        "Pods were evicted as a result.\n"
        "KUBECTL: kubectl describe node worker"
    )
    rc, kc = parse_diagnosis(text)
    assert "disk space" in rc
    assert "evicted" in rc.lower()
    assert kc == "kubectl describe node worker"


# ── Calidad: concisión y saneo del kubectl ──────────────────────────────────

@pytest.mark.unit
def test_parse_strips_tutorial_rambling():
    """Una salida tipo tutorial (listas, markdown, pasos) se reduce a frases concisas."""
    text = (
        "El primer paso para investigar este problema sería un análisis detallado. "
        "Aquí te muestro cómo hacerlo:\n"
        "1. **Obtener el nombre del volumen**:\n```bash\n```\n"
        "2. **Verificar las propiedades**: Size, AccessModes, etc.\n"
        "3. Más pasos y más pasos y más texto interminable que no aporta nada."
    )
    rc, kc = parse_diagnosis(text)
    assert "```" not in rc and "**" not in rc and "1." not in rc
    assert len(rc) <= 330
    assert rc.count(".") <= 4  # pocas frases


@pytest.mark.unit
def test_parse_removes_analysis_preamble():
    rc, _ = parse_diagnosis("Analysis The pod is failing due to OOM.")
    assert not rc.lower().startswith("analysis")


@pytest.mark.unit
def test_sanitize_kubectl_multi_namespace():
    assert sanitize_kubectl("kubectl describe pvc -n aiops-demo, longhorn-system, postgresql") == \
        "kubectl describe pvc -n aiops-demo"


@pytest.mark.unit
def test_sanitize_kubectl_strips_pipe_tail():
    assert sanitize_kubectl("kubectl get pods -n prod | grep Error | awk '{print $1}'") == \
        "kubectl get pods -n prod"


@pytest.mark.unit
def test_sanitize_kubectl_node_has_no_namespace():
    # node es cluster-scoped: no debe llevar -n
    out = sanitize_kubectl("kubectl describe node -n aiops-demo")
    assert "-n" not in out
    assert out == "kubectl describe node"


@pytest.mark.unit
def test_sanitize_kubectl_valid_unchanged():
    cmd = "kubectl logs api -n prod --tail=20"
    assert sanitize_kubectl(cmd) == cmd


@pytest.mark.unit
def test_sanitize_kubectl_placeholder_to_default():
    assert sanitize_kubectl("kubectl logs <nombre-pod> -n prod") == _DEFAULT_KUBECTL
    assert sanitize_kubectl("kubectl describe pod <pod> -n <ns>") == _DEFAULT_KUBECTL


@pytest.mark.unit
def test_sanitize_kubectl_strips_stray_backtick():
    assert sanitize_kubectl("kubectl describe pvc` -n postgresql") == "kubectl describe pvc -n postgresql"


@pytest.mark.unit
def test_sanitize_kubectl_non_kubectl_to_default():
    assert sanitize_kubectl("describe the pvc in postgresql namespace") == _DEFAULT_KUBECTL


@pytest.mark.unit
def test_parse_diagnosis_applies_sanitize():
    rc, kc = parse_diagnosis("ROOT CAUSE: x\nKUBECTL: kubectl describe pvc -n a, b, c")
    assert kc == "kubectl describe pvc -n a"


# ── build_event_sample: acotado ─────────────────────────────────────────────

@pytest.mark.unit
def test_build_event_sample_truncates_long_lines():
    logs = ["x" * 1000]
    text, n = build_event_sample(logs, max_logs=40)
    assert n == 1
    assert "…" in text          # línea truncada
    assert len(text) < 300      # muy por debajo de la línea original


@pytest.mark.unit
def test_build_event_sample_caps_total_size():
    logs = [f"línea de evento número {i} con algo de texto" for i in range(500)]
    text, n = build_event_sample(logs, max_logs=200)
    # El total queda acotado para no reventar num_ctx
    assert len(text) <= 3500 + 10


@pytest.mark.unit
def test_build_event_sample_keeps_last_n():
    logs = [f"e{i}" for i in range(100)]
    text, n = build_event_sample(logs, max_logs=5)
    assert n == 5
    assert "e99" in text
    assert "e0\n" not in text


@pytest.mark.unit
def test_window_event_sample_prefers_error_logs():
    """Si hay suficientes logs de error, la muestra del RCA lidera con ellos."""
    w = SimpleNamespace(
        raw_logs=["normal log %d" % i for i in range(50)],
        error_logs=["FATAL: the cluster is on fire %d" % i for i in range(10)],
    )
    text, n, label = window_event_sample(w, max_logs=40)
    assert "on fire" in text
    assert "error" in label.lower()
    assert "normal log" not in text  # lidera con errores, no logs normales


@pytest.mark.unit
def test_window_event_sample_falls_back_to_raw():
    w = SimpleNamespace(raw_logs=["normal log %d" % i for i in range(5)], error_logs=[])
    text, n, label = window_event_sample(w, max_logs=40)
    assert "normal log" in text
    assert label == "Event sample"


# ── cluster_error_templates: densa la señal por plantilla ────────────────────

def _erec(raw, template, cluster_id, namespace="pg"):
    return SimpleNamespace(raw=raw, template=template, cluster_id=cluster_id, namespace=namespace)


@pytest.mark.unit
def test_cluster_groups_and_counts_by_template():
    """12 errores de la misma plantilla → una línea '12× plantilla', no 12 líneas."""
    recs = [_erec(f'FATAL: role "user{i}" does not exist',
                  'FATAL: role "<*>" does not exist', 7) for i in range(12)]
    text, distinct = ollama_rca.cluster_error_templates(recs)
    assert distinct == 1
    assert "12×" in text
    assert 'role "<*>" does not exist' in text
    # un ejemplo real concreto acompaña a la plantilla
    assert "ej:" in text and "user0" in text
    # NO repite las 12 líneas crudas
    assert text.count("does not exist") <= 3


@pytest.mark.unit
def test_cluster_orders_by_frequency():
    recs = [_erec("conn refused", "conn refused", 2) for _ in range(2)]
    recs += [_erec(f"role {i} missing", "role <*> missing", 7) for i in range(9)]
    text, distinct = ollama_rca.cluster_error_templates(recs)
    assert distinct == 2
    # la plantilla más frecuente (9×) aparece antes que la de 2×
    assert text.index("9×") < text.index("2×")


@pytest.mark.unit
def test_cluster_labels_namespace():
    recs = [_erec("boom", "boom", 1, namespace="longhorn-system") for _ in range(6)]
    text, distinct = ollama_rca.cluster_error_templates(recs)
    assert "[longhorn-system]" in text


@pytest.mark.unit
def test_cluster_separates_same_template_different_namespace():
    recs = [_erec("x", "boom <*>", 1, namespace="a") for _ in range(3)]
    recs += [_erec("y", "boom <*>", 1, namespace="b") for _ in range(3)]
    _text, distinct = ollama_rca.cluster_error_templates(recs)
    assert distinct == 2  # mismo patrón pero distinto namespace = dos grupos


@pytest.mark.unit
def test_window_event_sample_uses_clustering_when_records_present():
    recs = [_erec(f'FATAL: role "u{i}" does not exist',
                  'FATAL: role "<*>" does not exist', 7) for i in range(8)]
    w = SimpleNamespace(
        raw_logs=["noise"] * 8,
        error_logs=[r.raw for r in recs],
        error_records=recs,
    )
    text, n, label = window_event_sample(w, max_logs=40)
    assert "8×" in text
    assert "pattern" in label.lower() or "template" in label.lower()
    assert n == 8  # nº total de líneas de error


@pytest.mark.unit
def test_window_event_sample_focuses_on_primary_namespace():
    """La evidencia se filtra al namespace culpable dominante, no a todos."""
    recs = [_erec(f"role {i} missing", "role <*> missing", 7, namespace="postgresql")
            for i in range(9)]
    recs += [_erec("volume degraded", "volume degraded", 8, namespace="longhorn-system")
             for _ in range(2)]
    w = SimpleNamespace(
        raw_logs=["noise"] * 11,
        error_logs=[r.raw for r in recs],
        error_records=recs,
        primary_namespace="postgresql",
    )
    text, n, label = window_event_sample(w, max_logs=40)
    assert "role <*> missing" in text
    assert "longhorn" not in text          # el namespace secundario se excluye
    assert "postgresql" in label.lower()   # la etiqueta nombra el foco
    assert n == 9                          # solo las líneas del primario


# ── rca_focus / rca_namespaces_line: liderar el prompt con el culpable ───────

@pytest.mark.unit
def test_rca_focus_primary_and_others():
    w = SimpleNamespace(primary_namespace="postgresql",
                        focus_namespaces=["argocd", "longhorn-system", "postgresql"])
    primary, others = ollama_rca.rca_focus(w)
    assert primary == "postgresql"
    assert set(others) == {"argocd", "longhorn-system"}


@pytest.mark.unit
def test_rca_focus_fallback_to_first_when_no_primary():
    w = SimpleNamespace(primary_namespace=None, focus_namespaces=["argocd", "kube-system"])
    primary, others = ollama_rca.rca_focus(w)
    assert primary == "argocd"
    assert others == ["kube-system"]


@pytest.mark.unit
def test_rca_namespaces_line_mentions_others_as_context():
    w = SimpleNamespace(primary_namespace="postgresql",
                        focus_namespaces=["postgresql", "longhorn-system"])
    line = ollama_rca.rca_namespaces_line(w)
    assert "postgresql" in line
    assert "longhorn-system" in line
    assert line.index("postgresql") < line.index("longhorn-system")  # foco primero


# ── Punto 3: anti-deriva del SLM + fallback determinista ─────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("bad", [
    "Lo siento, parece que hay un error en la estructura de tu mensaje.",
    "Here are some additional steps you can take: Step 4: Check Node Status",
    "En respuesta a la primera revisión: El primer paso para investigar...",
    "I'm sorry, but as an AI I cannot determine the cause.",
    "Parece que falta información importante para continuar.",
    "",
])
def test_looks_like_drift_detects_garbage(bad):
    assert ollama_rca._looks_like_drift(bad) is True


@pytest.mark.unit
def test_looks_like_drift_accepts_good_diagnosis():
    good = "El rol de PostgreSQL $(POSTGRES_USER) no existe, lo que impide arrancar los pods."
    assert ollama_rca._looks_like_drift(good) is False


def _win_with_errors(primary="postgresql"):
    recs = [_erec(f'FATAL: role "u{i}" does not exist',
                  'FATAL: role "<*>" does not exist', 7, namespace=primary) for i in range(7)]
    recs += [_erec("conn refused", "could not connect <*>", 8, namespace=primary) for _ in range(2)]
    return SimpleNamespace(error_records=recs, primary_namespace=primary,
                           focus_namespaces=[primary])


@pytest.mark.unit
def test_synthesize_root_cause_from_templates():
    """Sin un buen diagnóstico del modelo, se sintetiza desde la plantilla dominante."""
    rc = ollama_rca.synthesize_root_cause(_win_with_errors())
    assert "postgresql" in rc
    assert "7" in rc                               # frecuencia del patrón dominante
    assert 'role "<*>" does not exist' in rc       # la plantilla real
    assert ollama_rca._looks_like_drift(rc) is False


@pytest.mark.unit
def test_ensure_meaningful_replaces_drift_with_synthesis():
    w = _win_with_errors()
    out = ollama_rca.ensure_meaningful_root_cause("Lo siento, no puedo continuar.", w)
    assert "postgresql" in out
    assert "lo siento" not in out.lower()


@pytest.mark.unit
def test_ensure_meaningful_keeps_good_diagnosis():
    w = _win_with_errors()
    good = "El rol de PostgreSQL no existe e impide el arranque."
    assert ollama_rca.ensure_meaningful_root_cause(good, w) == good


@pytest.mark.unit
def test_ensure_meaningful_no_records_falls_back_to_text():
    """Sin error_records no se inventa nada: se conserva el texto original."""
    w = SimpleNamespace(error_records=[], primary_namespace=None, focus_namespaces=[])
    out = ollama_rca.ensure_meaningful_root_cause("Sin causa raíz determinable.", w)
    assert out == "Sin causa raíz determinable."


# ── diagnose() con red mockeada ─────────────────────────────────────────────

@dataclass
class _W:
    index: int = 1
    raw_logs: list = field(default_factory=lambda: ["evento a", "evento b"])
    namespaces: set = field(default_factory=lambda: {"default"})
    start_time: float = 0.0
    end_time: float = 60.0
    log_count: int = 2
    template_count: int = 2

    @property
    def focus_namespaces(self):
        return sorted(self.namespaces)


@dataclass
class _Scored:
    window: _W = field(default_factory=_W)
    score: float = 0.88
    model_version: int = 1


class _FakeResp:
    def __init__(self, content):
        self._content = content
    def raise_for_status(self):
        pass
    def json(self):
        return {"message": {"content": self._content}}


class _FakeClient:
    def __init__(self, content):
        self._content = content
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def post(self, url, json=None):
        return _FakeResp(self._content)


@pytest.mark.unit
def test_diagnose_returns_parsed_result(monkeypatch):
    content = "ROOT CAUSE: Nodo sin memoria\nKUBECTL: kubectl describe node worker"
    monkeypatch.setattr(ollama_rca.httpx, "Client", lambda *a, **k: _FakeClient(content))
    res = OllamaRCA().diagnose(_Scored())
    assert res.root_cause == "Nodo sin memoria"
    assert res.kubectl_command == "kubectl describe node worker"
    assert res.mode == "single_shot"
