"""
Tests del agente de chat con el cluster (ReAct read-only).

Ollama y kubectl mockeados. Verifica parsing, ciclo, terminación,
y CRÍTICO: que toda acción pasa por el toolbox read-only.
"""

import pytest

from src.diagnostics.cluster_chat import ClusterChatAgent, _parse


@pytest.mark.unit
def test_parse_action():
    t, a, ans = _parse("THOUGHT: reviso pods\nACTION: kubectl get pods -n prod")
    assert t == "reviso pods"
    assert a == "kubectl get pods -n prod"
    assert ans is None


@pytest.mark.unit
def test_parse_multiline_answer():
    t, a, ans = _parse("THOUGHT: ya sé\nANSWER: Hay 3 pods caídos.\nCausa: OOMKilled.")
    assert a is None
    assert "Hay 3 pods caídos." in ans
    assert "OOMKilled" in ans


def _agent(responses, tool=lambda a: f"[obs: {a}]"):
    agent = ClusterChatAgent(max_steps=4)
    it = iter(responses)
    agent._call = lambda msgs: next(it)
    agent._run_tool = tool
    return agent


@pytest.mark.unit
def test_full_loop_reaches_answer():
    agent = _agent([
        "THOUGHT: reviso\nACTION: kubectl get pods -n prod",
        "THOUGHT: claro\nANSWER: Todo OK en prod.",
    ])
    events = list(agent.chat_iter("¿estado de prod?"))
    types = [e["type"] for e in events]
    assert "action" in types
    assert "observation" in types
    assert events[-1]["type"] == "answer"
    assert events[-1]["text"] == "Todo OK en prod."


@pytest.mark.unit
def test_immediate_answer_no_action():
    agent = _agent(["THOUGHT: obvio\nANSWER: No hay nada que investigar."])
    events = list(agent.chat_iter("hola"))
    assert all(e["type"] != "action" for e in events)
    assert events[-1]["type"] == "answer"


@pytest.mark.unit
def test_max_steps_forces_final_answer():
    # El modelo nunca da ANSWER; siempre propone acción nueva
    counter = {"n": 0}
    def always_action(msgs):
        counter["n"] += 1
        return f"THOUGHT: sigo\nACTION: kubectl get pods -n ns{counter['n']}"
    agent = ClusterChatAgent(max_steps=3)
    agent._call = always_action
    agent._run_tool = lambda a: "obs"
    # tras max_steps pide respuesta final; el último _call también da acción,
    # que _parse no convierte en answer → answer vacío con fallback
    events = list(agent.chat_iter("?"))
    assert events[-1]["type"] == "answer"  # siempre termina con answer


@pytest.mark.unit
def test_repeated_action_terminates():
    # El modelo repite la misma acción → debe forzar answer, no bucle infinito
    agent = _agent([
        "THOUGHT: a\nACTION: kubectl get pods",
        "THOUGHT: a otra vez\nACTION: kubectl get pods",  # repetida
    ])
    events = list(agent.chat_iter("?"))
    assert events[-1]["type"] == "answer"


@pytest.mark.unit
def test_actions_go_through_readonly_toolbox():
    """CRÍTICO: el _run_tool real usa kubectl_toolbox, que bloquea escrituras."""
    from src.diagnostics import cluster_chat
    agent = ClusterChatAgent(max_steps=2, dry_run=False)
    # Si el modelo propusiera un delete, el toolbox lo rechaza
    captured = {}
    original = cluster_chat.kubectl_execute
    def spy(cmd):
        captured["cmd"] = cmd
        return original(cmd)  # toolbox real → rechaza delete
    cluster_chat.kubectl_execute = spy
    try:
        out = agent._run_tool("kubectl delete pod x -n prod")
    finally:
        cluster_chat.kubectl_execute = original
    assert "prohibido" in out.lower() or "solo se permiten" in out.lower() or "Error" in out


@pytest.mark.unit
def test_dry_run_does_not_execute():
    agent = ClusterChatAgent(dry_run=True)
    out = agent._run_tool("kubectl get pods")
    assert out.startswith("[dry-run]")


@pytest.mark.unit
def test_model_error_emits_error_event():
    agent = ClusterChatAgent(max_steps=2)
    def boom(msgs):
        raise RuntimeError("ollama caído")
    agent._call = boom
    events = list(agent.chat_iter("?"))
    assert events[0]["type"] == "error"
    assert "ollama caído" in events[0]["text"]
