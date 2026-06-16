"""
Tests de la capa RCA single-shot y utilidades compartidas (src/diagnostics/ollama_rca.py).

Cubre el parseo tolerante (parse_diagnosis), el acotado de la muestra de eventos
(build_event_sample — el fix del bug 'Could not parse root cause') y diagnose()
con la red mockeada.
"""

from dataclasses import dataclass, field

import pytest

from src.diagnostics import ollama_rca
from src.diagnostics.ollama_rca import (
    _DEFAULT_KUBECTL,
    OllamaRCA,
    build_event_sample,
    parse_diagnosis,
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
