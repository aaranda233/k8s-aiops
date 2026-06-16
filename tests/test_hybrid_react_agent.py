"""
Tests del agente híbrido de RCA (src/diagnostics/hybrid_react_agent.py).

Verifica el parseo del investigador, el ciclo de investigación (fase 1) con
terminación por DONE / acción repetida, y la construcción del DiagnosisResult
con la síntesis del experto (fase 2) mockeada.
"""

from dataclasses import dataclass, field

import pytest

from src.diagnostics.hybrid_react_agent import HybridReActAgent, _parse_investigator


@dataclass
class _W:
    index: int = 7
    raw_logs: list = field(default_factory=lambda: ["evento de cluster"])
    namespaces: set = field(default_factory=lambda: {"llm-app"})
    start_time: float = 0.0
    end_time: float = 60.0
    log_count: int = 545
    template_count: int = 12


@dataclass
class _Scored:
    window: _W = field(default_factory=_W)
    score: float = 0.95
    model_version: int = 2


@pytest.mark.unit
def test_parse_investigator_action():
    t, a, done = _parse_investigator("THOUGHT: reviso\nACTION: kubectl get pods -A")
    assert t == "reviso"
    assert a == "kubectl get pods -A"
    assert done is False


@pytest.mark.unit
def test_parse_investigator_done():
    t, a, done = _parse_investigator("THOUGHT: suficiente\nDONE")
    assert done is True
    assert a is None


@pytest.mark.unit
def test_diagnose_builds_hybrid_result():
    agent = HybridReActAgent(max_steps=2, dry_run=True)
    # Fase 1 (investigador) y fase 2 (experto con grammar) mockeadas
    agent._call = lambda messages, model, num_predict=300: "THOUGHT: miro\nACTION: kubectl get pods -A"
    agent._call_expert_with_grammar = lambda content: (
        "Pod oauth2-proxy-vllm en CrashLoopBackOff por OIDC 404",
        "kubectl logs oauth2-proxy-vllm -n llm-app",
    )
    res = agent.diagnose(_Scored())
    assert res.mode == "hybrid"
    assert "CrashLoopBackOff" in res.root_cause
    assert res.kubectl_command == "kubectl logs oauth2-proxy-vllm -n llm-app"
    assert res.steps_taken >= 1


@pytest.mark.unit
def test_investigate_stops_on_done():
    agent = HybridReActAgent(max_steps=5, dry_run=True)
    agent._call = lambda messages, model, num_predict=300: "THOUGHT: ya está\nDONE"
    agent._call_expert_with_grammar = lambda content: ("causa", "kubectl get pods")
    res = agent.diagnose(_Scored())
    # Un único paso (DONE inmediato)
    assert res.steps_taken == 1


@pytest.mark.unit
def test_investigate_stops_on_repeated_action():
    agent = HybridReActAgent(max_steps=5, dry_run=True)
    agent._call = lambda messages, model, num_predict=300: "THOUGHT: miro\nACTION: kubectl get pods -A"
    agent._call_expert_with_grammar = lambda content: ("causa", "kubectl get pods")
    res = agent.diagnose(_Scored())
    # La acción repetida corta el ciclo (no agota los 5 pasos)
    assert res.steps_taken <= 2


@pytest.mark.unit
def test_dry_run_tool_does_not_execute():
    agent = HybridReActAgent(dry_run=True)
    assert agent._run_tool("kubectl get pods").startswith("[dry-run]")
