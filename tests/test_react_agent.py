"""
Tests del agente ReAct de RCA (src/diagnostics/react_agent.py).

Ollama y kubectl mockeados. Verifica el parseo del formato ReAct, el ciclo
THOUGHT→ACTION→OBSERVATION→FINAL, terminación por acción repetida y por
máximo de pasos, y la construcción del DiagnosisResult.
"""

from dataclasses import dataclass, field

import pytest

from src.diagnostics.react_agent import ReActAgent, _parse_response


@dataclass
class _W:
    index: int = 1
    raw_logs: list = field(default_factory=lambda: ["evento"])
    namespaces: set = field(default_factory=lambda: {"default"})
    start_time: float = 0.0
    end_time: float = 60.0
    log_count: int = 10
    template_count: int = 3

    @property
    def focus_namespaces(self):
        return sorted(self.namespaces)


@dataclass
class _Scored:
    window: _W = field(default_factory=_W)
    score: float = 0.9
    model_version: int = 1


@pytest.mark.unit
def test_parse_action_format():
    t, a, fin, rc, kc, conf = _parse_response("THOUGHT: reviso\nACTION: kubectl get pods -n default")
    assert t == "reviso"
    assert a == "kubectl get pods -n default"
    assert fin is False


@pytest.mark.unit
def test_parse_final_format():
    text = ("THOUGHT: claro\nFINAL:\nROOT CAUSE: OOMKilled en api\n"
            "KUBECTL: kubectl describe pod api\nCONFIDENCE: high")
    t, a, fin, rc, kc, conf = _parse_response(text)
    assert fin is True
    assert rc == "OOMKilled en api"
    assert kc == "kubectl describe pod api"
    assert conf == "high"


@pytest.mark.unit
def test_diagnose_reaches_final():
    agent = ReActAgent(max_steps=4, dry_run=True)
    responses = iter([
        "THOUGHT: investigo\nACTION: kubectl get pods -n default",
        "THOUGHT: ya lo veo\nFINAL:\nROOT CAUSE: Pod en CrashLoopBackOff\n"
        "KUBECTL: kubectl logs api -n default\nCONFIDENCE: high",
    ])
    agent._call_llm = lambda msgs: next(responses)
    res = agent.diagnose(_Scored())
    assert res.mode == "react"
    assert res.root_cause == "Pod en CrashLoopBackOff"
    assert res.kubectl_command == "kubectl logs api -n default"
    assert res.confidence == "high"
    assert any(s.action for s in res.react_trace)


@pytest.mark.unit
def test_diagnose_repeated_action_breaks():
    agent = ReActAgent(max_steps=5, dry_run=True)
    agent._call_llm = lambda msgs: "THOUGHT: miro\nACTION: kubectl get pods"
    # La acción se repite → en la 2ª iteración entra a la rama de cierre
    res = agent.diagnose(_Scored())
    assert res.mode == "react"
    assert res.steps_taken >= 1


@pytest.mark.unit
def test_diagnose_max_steps_forces_final_call():
    calls = {"n": 0}
    def call(msgs):
        calls["n"] += 1
        # Siempre nueva acción distinta para agotar los pasos
        return f"THOUGHT: paso\nACTION: kubectl get pods -n ns{calls['n']}"
    agent = ReActAgent(max_steps=2, dry_run=True)
    agent._call_llm = call
    res = agent.diagnose(_Scored())
    # Tras agotar pasos, se hace una llamada final extra
    assert calls["n"] == 3  # 2 pasos + 1 final
    assert res.mode == "react"


@pytest.mark.unit
def test_dry_run_tool_does_not_execute():
    agent = ReActAgent(dry_run=True)
    assert agent._run_tool("kubectl get pods").startswith("[dry-run]")
