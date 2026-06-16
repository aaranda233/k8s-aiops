"""
Tests del agente de chat con el cluster (ReAct read-only).

Ollama y kubectl mockeados. Verifica parsing, ciclo, terminación,
y CRÍTICO: que toda acción pasa por el toolbox read-only.
"""

import pytest

from src.diagnostics.cluster_chat import (
    ClusterChatAgent,
    _cluster_digest,
    _describe_cmd,
    _name_focus,
    _parse,
    _parse_pod_rows,
    _top_problem,
    extract_problem_pods,
)

# Pods con varios 'parser' (todos sanos) repartidos entre los rotos.
_PODS_WITH_PARSERS = """\
NAMESPACE   NAME                                 READY   STATUS             RESTARTS  AGE
default     anecoop-parser-f9f477478-l6g6x       1/1     Running            0         41d
default     edeka-parser-58d574d8d4-2jz9x        1/1     Running            0         41d
default     eurogroup-parser-6dfd6df857-6qxk9    1/1     Running            0         28d
default     iberiana-parser-8948b99bf-gwq28      1/1     Running            0         17h
llm-app     oauth2-proxy-vllm-56ffbf5d4d-vkhnb   0/1     CrashLoopBackOff   1624      5d
"""

# Muestra realista de `kubectl get pods -A` con problemas claros enterrados
# entre muchos pods sanos (reproduce el fallo de truncado a 600 chars).
_PODS_SAMPLE = """\
NAMESPACE          NAME                                  READY   STATUS                   RESTARTS          AGE
aeat-retenciones   aeat-retenciones-5b95985bc9-bg5p5     1/1     Running                  0                 34d
argocd             argocd-server-db995fb4d-z5qfp         1/1     Running                  0                 59d
banca-conection    banca-sync-1-29678640-dj9zg           0/1     Error                    0                 10d
banca-conection    banca-sync-1-29690220-qnfx5           0/1     Completed                0                 2d2h
default            curl-test-master                      0/1     ContainerStatusUnknown   0                 124d
default            dashboard-644cc689c4-2gttn            1/1     Running                  0                 7d17h
llm-app            oauth2-proxy-vllm-56ffbf5d4d-vkhnb    0/1     CrashLoopBackOff         1620 (3m34s ago)  5d17h
kube-system        kube-scheduler-ubuntumaster           1/1     Running                  28 (44d ago)      290d
"""


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
def test_clean_cmd_strips_duplicated_namespace_prefix():
    """El modelo escribe ns/name como recurso mientras ya pasa -n ns; se normaliza."""
    _, a, _ = _parse("ACTION: kubectl describe pod default/node-debugger -n default")
    assert a == "kubectl describe pod node-debugger -n default"
    _, a, _ = _parse("ACTION: kubectl logs llm-app/vllm-proxy -n llm-app --tail=40")
    assert a == "kubectl logs vllm-proxy -n llm-app --tail=40"


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
    # El triage determinista ocurre antes de llamar al modelo; cuando el modelo
    # falla en la fase de profundización debe emitirse un evento de error.
    agent = ClusterChatAgent(max_steps=2)
    agent._run_tool = lambda a: "(sin pods)"
    def boom(msgs, model=None):
        raise RuntimeError("ollama caído")
    agent._call = boom
    events = list(agent.chat_iter("?"))
    errors = [e for e in events if e["type"] == "error"]
    assert errors
    assert "ollama caído" in errors[0]["text"]


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


# ── Análisis determinista de pods ──────────────────────────────────────────

@pytest.mark.unit
def test_extract_problem_pods_finds_failures_ignores_completed():
    problems = extract_problem_pods(_PODS_SAMPLE)
    names = {p["name"] for p in problems}
    # Fallos reales detectados
    assert "oauth2-proxy-vllm-56ffbf5d4d-vkhnb" in names   # CrashLoopBackOff
    assert "banca-sync-1-29678640-dj9zg" in names          # Error
    assert "curl-test-master" in names                     # ContainerStatusUnknown
    # Jobs terminados y pods sanos NO son problemas
    assert "banca-sync-1-29690220-qnfx5" not in names      # Completed
    assert "aeat-retenciones-5b95985bc9-bg5p5" not in names
    assert "kube-scheduler-ubuntumaster" not in names      # Running 1/1 (reinicios viejos)


@pytest.mark.unit
def test_extract_running_but_not_ready_is_problem():
    out = "NAMESPACE NAME READY STATUS RESTARTS AGE\nns app-x 0/2 Running 0 3d\n"
    problems = extract_problem_pods(out)
    assert len(problems) == 1
    assert problems[0]["name"] == "app-x"


@pytest.mark.unit
def test_cluster_digest_includes_failures_not_truncated():
    """REGRESIÓN: el resumen debe contener los fallos aunque estén al final."""
    digest = _cluster_digest(_PODS_SAMPLE)
    assert "CrashLoopBackOff" in digest
    assert "oauth2-proxy-vllm" in digest
    assert "con problemas" in digest
    # El experto recibiría esto en vez de los primeros 600 chars sin fallos.


@pytest.mark.unit
def test_cluster_digest_healthy_says_so():
    healthy = ("NAMESPACE NAME READY STATUS RESTARTS AGE\n"
               "ns a 1/1 Running 0 3d\nns b 2/2 Running 0 3d\n")
    digest = _cluster_digest(healthy)
    assert "sano" in digest.lower()
    assert "0 con problemas" in digest


@pytest.mark.unit
def test_top_problem_prioritizes_crashloop():
    problems = extract_problem_pods(_PODS_SAMPLE)
    top = _top_problem(problems)
    assert top["status"] == "CrashLoopBackOff"
    assert _describe_cmd(top) == (
        "kubectl describe pod oauth2-proxy-vllm-56ffbf5d4d-vkhnb -n llm-app"
    )


# ── Flujo del agente con triage determinista ───────────────────────────────

@pytest.mark.unit
def test_triage_is_deterministic_first_action():
    """El primer comando ejecutado SIEMPRE es el triage, sin depender del modelo."""
    tool_calls = []
    agent = ClusterChatAgent(max_steps=3)
    agent._call = lambda msgs, model=None: "Diagnóstico." if model == agent.expert_model \
        else "THOUGHT: ya\nANSWER: listo"
    def tool(a):
        tool_calls.append(a)
        return _PODS_SAMPLE
    agent._run_tool = tool
    list(agent.chat_iter("¿qué pasa?"))
    assert tool_calls[0] == "kubectl get pods -A"


@pytest.mark.unit
def test_no_dithering_auto_drills_when_model_wont_act():
    """Si el modelo solo divaga (THOUGHT sin ACTION), el harness profundiza solo."""
    tool_calls = []
    agent = ClusterChatAgent(max_steps=4)
    def call(msgs, model=None):
        if model == agent.expert_model:
            return "El proxy vllm está en CrashLoopBackOff."
        return "THOUGHT: necesito mirar los pods"  # nunca emite ACTION
    agent._call = call
    def tool(a):
        tool_calls.append(a)
        return _PODS_SAMPLE if a == "kubectl get pods -A" else "Events: Back-off restarting"
    agent._run_tool = tool
    list(agent.chat_iter("¿qué pasa?"))
    # Auto-profundizó en el pod más crítico pese a que el modelo no dio ACTION
    assert any("describe pod oauth2-proxy-vllm" in c for c in tool_calls)


@pytest.mark.unit
def test_scoped_question_queries_mentioned_namespace():
    """Si la pregunta menciona un namespace real, se consulta ese namespace."""
    tool_calls = []
    agent = ClusterChatAgent(max_steps=3)
    agent._call = lambda msgs, model=None: "Respuesta." if model == agent.expert_model \
        else "THOUGHT: ya\nANSWER: listo"
    def tool(a):
        tool_calls.append(a)
        return _PODS_SAMPLE
    agent._run_tool = tool
    list(agent.chat_iter("cuantos pods hay en el namespace argocd"))
    assert "kubectl get pods -n argocd" in tool_calls


@pytest.mark.unit
def test_no_scoped_query_without_namespace_mention():
    tool_calls = []
    agent = ClusterChatAgent(max_steps=3)
    agent._call = lambda msgs, model=None: "Respuesta." if model == agent.expert_model \
        else "THOUGHT: ya\nANSWER: listo"
    def tool(a):
        tool_calls.append(a)
        return _PODS_SAMPLE
    agent._run_tool = tool
    list(agent.chat_iter("hay algun problema en el cluster"))
    assert not any(c.startswith("kubectl get pods -n ") for c in tool_calls)


@pytest.mark.unit
def test_name_focus_matches_pods_by_name_keyword():
    rows = _parse_pod_rows(_PODS_WITH_PARSERS)
    kw, matched = _name_focus("¿cómo están los pods del parser?", rows)
    assert kw == "parser"
    names = {m["name"] for m in matched}
    assert len(matched) == 4
    assert "anecoop-parser-f9f477478-l6g6x" in names
    assert "oauth2-proxy-vllm-56ffbf5d4d-vkhnb" not in names  # no es un parser


@pytest.mark.unit
def test_name_focus_returns_none_without_match():
    rows = _parse_pod_rows(_PODS_WITH_PARSERS)
    kw, matched = _name_focus("¿hay algún problema en el cluster?", rows)
    assert kw is None
    assert matched == []


@pytest.mark.unit
def test_chat_focuses_on_parser_pods_in_evidence():
    """REGRESIÓN: preguntar por 'parser' debe enfocar esos pods, no el peor del cluster."""
    captured = {}
    tool_calls = []
    agent = ClusterChatAgent(max_steps=3)
    def call(msgs, model=None):
        if model == agent.expert_model:
            captured["evidence"] = msgs[-1]["content"]
            return "Los 4 pods parser están Running."
        return "THOUGHT: ya\nANSWER: listo"
    agent._call = call
    def tool(a):
        tool_calls.append(a)
        return _PODS_WITH_PARSERS
    agent._run_tool = tool
    list(agent.chat_iter("¿cómo están los pods del parser?"))
    # La evidencia que llega al experto contiene los pods parser
    assert "anecoop-parser" in captured["evidence"]
    assert "parser" in captured["evidence"]
    # Y NO la lista de problemas ajenos del cluster (que confundía al experto)
    assert "CrashLoopBackOff" not in captured["evidence"]
    assert "oauth2-proxy-vllm" not in captured["evidence"]
    # No se auto-profundizó en el vllm (la pregunta no era sobre el peor pod)
    assert not any("describe pod oauth2-proxy-vllm" in c for c in tool_calls)


@pytest.mark.unit
def test_synthesis_receives_failures_in_evidence():
    """La evidencia que llega al experto contiene los fallos (no truncados)."""
    captured = {}
    agent = ClusterChatAgent(max_steps=3)
    def call(msgs, model=None):
        if model == agent.expert_model:
            captured["evidence"] = msgs[-1]["content"]
            return "Diagnóstico final."
        return "THOUGHT: ya tengo la causa\nANSWER: listo"
    agent._call = call
    agent._run_tool = lambda a: _PODS_SAMPLE
    list(agent.chat_iter("¿qué pasa en el cluster?"))
    assert "CrashLoopBackOff" in captured["evidence"]
    assert "oauth2-proxy-vllm" in captured["evidence"]


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
