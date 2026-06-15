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


@pytest.mark.unit
def test_parse_lenient_kubectl_without_action_prefix():
    """El base a veces no usa 'ACTION:'; debe extraer el kubectl igual."""
    # En bloque de código markdown
    _, a, _ = _parse("THOUGHT: reviso el pod\n```\nkubectl describe pod sqldelete -n default\n```")
    assert a == "kubectl describe pod sqldelete -n default"
    # Como viñeta
    _, a, _ = _parse("Necesito ver los logs\n- kubectl logs sqldelete -n default")
    assert a == "kubectl logs sqldelete -n default"


@pytest.mark.unit
def test_parse_answer_takes_priority_over_loose_kubectl():
    # Si hay ANSWER, no debe confundir un kubectl mencionado en la respuesta
    _, a, ans = _parse("ANSWER: Ejecuta kubectl logs para ver más.")
    assert a is None
    assert "kubectl logs" in ans


def _agent(responses, tool=lambda a: f"[obs: {a}]", synth="Conclusión del experto."):
    """responses = lo que devuelve el INVESTIGADOR; synth = lo que devuelve el EXPERTO."""
    agent = ClusterChatAgent(max_steps=4)
    it = iter(responses)
    def call(msgs, model=None):
        # Si se llama con el modelo experto → es la síntesis final
        if model == agent.expert_model:
            return synth
        return next(it)
    agent._call = call
    agent._run_tool = tool
    return agent


@pytest.mark.unit
def test_full_loop_synthesizes_with_expert():
    agent = _agent([
        "THOUGHT: reviso\nACTION: kubectl get pods -n prod",
        "THOUGHT: claro\nANSWER: ok",
    ], synth="El pod api está OOMKilled, sube el límite de memoria.")
    events = list(agent.chat_iter("¿estado de prod?"))
    types = [e["type"] for e in events]
    assert "action" in types
    assert "observation" in types
    # La respuesta final la da el EXPERTO, no el investigador
    assert events[-1]["type"] == "answer"
    assert events[-1]["text"] == "El pod api está OOMKilled, sube el límite de memoria."


@pytest.mark.unit
def test_must_investigate_before_concluding():
    """Si el base intenta responder sin investigar, se le empuja a ejecutar kubectl."""
    agent = _agent([
        "THOUGHT: obvio\nANSWER: ya",                       # intento de responder sin evidencia
        "THOUGHT: vale, investigo\nACTION: kubectl get pods -n prod",
        "THOUGHT: listo\nANSWER: ok",
    ], synth="Respuesta con evidencia.")
    events = list(agent.chat_iter("hola"))
    # Acabó investigando (hay una acción) y la respuesta es del experto
    assert any(e["type"] == "action" for e in events)
    assert events[-1]["type"] == "answer"
    assert events[-1]["text"] == "Respuesta con evidencia."


@pytest.mark.unit
def test_max_steps_forces_expert_synthesis():
    counter = {"n": 0}
    agent = ClusterChatAgent(max_steps=3)
    def call(msgs, model=None):
        if model == agent.expert_model:
            return "Síntesis tras agotar pasos."
        counter["n"] += 1
        return f"THOUGHT: sigo\nACTION: kubectl get pods -n ns{counter['n']}"
    agent._call = call
    agent._run_tool = lambda a: "obs"
    events = list(agent.chat_iter("?"))
    assert events[-1]["type"] == "answer"
    assert events[-1]["text"] == "Síntesis tras agotar pasos."


@pytest.mark.unit
def test_repeated_action_terminates():
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
    def boom(msgs, model=None):
        raise RuntimeError("ollama caído")
    agent._call = boom
    events = list(agent.chat_iter("?"))
    assert events[0]["type"] == "error"
    assert "ollama caído" in events[0]["text"]


@pytest.mark.unit
def test_placeholder_command_is_rejected_not_executed():
    """Un comando con <placeholder> no debe ejecutarse; se corrige al modelo."""
    tool_calls = []
    agent = ClusterChatAgent(max_steps=4)
    responses = iter([
        "THOUGHT: miro el servicio\nACTION: kubectl get svc -n <namespace>",  # placeholder
        "THOUGHT: ahora con nombre real\nACTION: kubectl get pods -A",          # corregido
        "THOUGHT: listo\nANSWER: ok",
    ])
    def call(msgs, model=None):
        if model == agent.expert_model:
            return "Diagnóstico final."
        return next(responses)
    agent._call = call
    def tool(a):
        tool_calls.append(a)
        return "salida real"
    agent._run_tool = tool
    events = list(agent.chat_iter("¿qué pasa en producción?"))
    # el comando con placeholder NO se ejecutó
    assert "kubectl get svc -n <namespace>" not in tool_calls
    # el comando corregido sí
    assert "kubectl get pods -A" in tool_calls
    assert events[-1]["text"] == "Diagnóstico final."


@pytest.mark.unit
def test_synthesis_uses_expert_model():
    """La conclusión final debe pedirse al modelo experto, no al investigador."""
    agent = ClusterChatAgent(max_steps=2, model="base", expert_model="experto")
    used = []
    def call(msgs, model=None):
        used.append(model)
        if model == "experto":
            return "diagnóstico final"
        return "THOUGHT: miro\nACTION: kubectl get pods"
    agent._call = call
    agent._run_tool = lambda a: "obs"
    events = list(agent.chat_iter("?"))
    assert "experto" in used  # se invocó al experto para la síntesis
    assert events[-1]["text"] == "diagnóstico final"
