"""
Servidor FastAPI con WebSocket para la UI en tiempo real.

Arranca el pipeline en un hilo de fondo y emite eventos
a todos los clientes conectados via WebSocket.
"""

import asyncio
import json
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CollectorConfig, DetectorConfig, DiagnosticsConfig, PipelineConfig, RemediationConfig
from dataset.feedback_capture import record_feedback
from src.collector.topology_collector import TopologyCollector
from src.diagnostics.cluster_chat import ClusterChatAgent
from src.pipeline import AIOPsPipeline
from src.remediation.incident_log import IncidentLog
from src.remediation.incident_store import Incident, IncidentStore
from src.security.scanner import SecurityScanner
from web.event_bus import bus

# Logging a fichero (observabilidad independiente de cómo se lance el proceso).
# AIOPS_LOG_FILE=/ruta/al.log activa el FileHandler en el logger 'aiops.*'.
_log_file = os.getenv("AIOPS_LOG_FILE", "")
if _log_file:
    _fh = logging.FileHandler(_log_file)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _root = logging.getLogger("aiops")
    _root.addHandler(_fh)
    _root.setLevel(logging.INFO)

# Registro de incidentes compartido entre el pipeline (remediación) y la consola.
# Con log durable (persistencia + dataset de aprendizaje) y captura de feedback.
_incident_log = IncidentLog(os.getenv("AIOPS_INCIDENT_LOG", "data/incidents/incidents.jsonl"))
incident_store = IncidentStore(incident_log=_incident_log)
incident_store.set_feedback_hook(record_feedback)
# Rehidratar la consola con los incidentes persistidos en arranques previos.
for _snap in _incident_log.latest_incidents():
    try:
        incident_store._incidents[_snap["id"]] = Incident(**{
            k: v for k, v in _snap.items() if k in Incident.__dataclass_fields__
        })
    except Exception:
        pass

# Agente de chat read-only (investigación on-demand del cluster)
chat_agent = ClusterChatAgent(
    host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    model=os.getenv("REACT_BASE_MODEL", "qwen2.5:1.5b"),       # investigador
    expert_model=os.getenv("OLLAMA_MODEL", "k8s-rca-orpo:latest"),  # sintetiza la conclusión
    max_steps=int(os.getenv("CHAT_MAX_STEPS", "5")),
    dry_run=os.getenv("CHAT_DRY_RUN", "false").lower() == "true",
)

# Estado de la aplicación (para sondas de K8s)
_app_state = {"ready": False, "pipeline_thread": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Arranca el pipeline en un hilo de fondo al iniciar el servidor."""
    # Registrar el event loop del servidor en el bus
    bus.set_loop(asyncio.get_running_loop())

    # Configuración del pipeline — vía variables de entorno
    mode = os.getenv("PIPELINE_MODE", "live")
    bootstrap = int(os.getenv("BOOTSTRAP_WINDOWS", "5"))
    threshold = float(os.getenv("ANOMALY_THRESHOLD", "0.80"))
    window_size = float(os.getenv("WINDOW_SIZE", "60"))
    namespaces_env = os.getenv("NAMESPACES", "")
    namespaces = namespaces_env.split(",") if namespaces_env else None

    # Diagnóstico (RCA) configurable por env — necesario para que la remediación
    # (y el modo sombra) generen incidentes a partir de las anomalías detectadas.
    diag_enabled = os.getenv("DIAGNOSTICS_ENABLED", "false").lower() == "true"

    cfg = PipelineConfig(
        collector=CollectorConfig(
            namespaces=namespaces,
            window_size_seconds=window_size,
            bootstrap_windows=bootstrap,
        ),
        detector=DetectorConfig(anomaly_threshold=threshold),
        diagnostics=DiagnosticsConfig(enabled=diag_enabled),
        remediation=RemediationConfig(),
        replay_mode=(mode == "replay"),
    )

    def _run():
        # La creación del pipeline (conexión al cluster) va dentro del hilo:
        # si falla, el servidor HTTP sigue vivo y /ready reporta not_ready
        # en vez de hacer crash-loop del pod.
        try:
            # El incident_store es el mismo que usan los endpoints /api/incidents
            pipeline_instance = AIOPsPipeline(cfg=cfg, event_bus=bus, incident_store=incident_store)
            _app_state["ready"] = True
            if mode == "replay":
                pipeline_instance.run_replay()
            else:
                pipeline_instance.run_live()
        except Exception as e:
            logging.getLogger(__name__).error("Pipeline no pudo arrancar: %s", e)
            _app_state["ready"] = False

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _app_state["pipeline_thread"] = t

    yield
    # Apagado: el hilo es daemon, muere con el proceso. Nada que limpiar.


app = FastAPI(title="k8s-aiops", lifespan=lifespan)

# Servir ficheros estaticos
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ------------------------------------------------------------------
# WebSocket — un endpoint, multiples clientes
# ------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    q = bus.subscribe()
    try:
        while True:
            msg = await q.get()
            await ws.send_text(msg)
    except (WebSocketDisconnect, Exception):
        bus.unsubscribe(q)


# ------------------------------------------------------------------
# Estado de la aplicación (para sondas de K8s) — _app_state se define
# junto al lifespan, arriba.
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    """Liveness — el proceso responde. Usado por livenessProbe."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness — el pipeline está arrancado. Usado por readinessProbe."""
    thread = _app_state["pipeline_thread"]
    alive = thread is not None and thread.is_alive()
    if _app_state["ready"] and alive:
        return {"status": "ready"}
    return JSONResponse({"status": "not_ready", "pipeline_alive": alive}, status_code=503)


# ------------------------------------------------------------------
# Pagina principal
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


# ------------------------------------------------------------------
# Consola de incidencias — la decisión humana ocurre aquí
# ------------------------------------------------------------------

@app.get("/incidents", response_class=HTMLResponse)
async def incidents_page():
    html_path = Path(__file__).parent / "static" / "incidents.html"
    return HTMLResponse(html_path.read_text())


@app.get("/incidents/{incident_id}", response_class=HTMLResponse)
async def incident_detail_page(incident_id: str):
    # La misma SPA resuelve el detalle por id en el cliente
    html_path = Path(__file__).parent / "static" / "incidents.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/incidents")
async def api_list_incidents():
    return {"incidents": [i.to_dict() for i in incident_store.list()]}


@app.get("/api/incidents/{incident_id}")
async def api_get_incident(incident_id: str):
    inc = incident_store.get(incident_id)
    if inc is None:
        return JSONResponse({"error": "Incidente no encontrado"}, status_code=404)
    return inc.to_dict()


@app.post("/api/incidents/{incident_id}/approve")
async def api_approve(incident_id: str):
    if not incident_store.set_response(incident_id, "approved"):
        return JSONResponse({"error": "Incidente no encontrado"}, status_code=404)
    return {"status": "approved", "id": incident_id}


@app.post("/api/incidents/{incident_id}/reject")
async def api_reject(incident_id: str):
    if not incident_store.set_response(incident_id, "rejected"):
        return JSONResponse({"error": "Incidente no encontrado"}, status_code=404)
    return {"status": "rejected", "id": incident_id}


@app.post("/api/incidents/{incident_id}/correct")
async def api_correct(incident_id: str, correction: dict):
    """Corrección humana del diagnóstico (señal de aprendizaje de máxima calidad).

    Body: {"root_cause": "...", "kubectl": "..."}. NO ejecuta nada: solo guarda la
    corrección para el dataset de feedback.
    """
    inc = incident_store.get(incident_id)
    if inc is None:
        return JSONResponse({"error": "Incidente no encontrado"}, status_code=404)
    rc = (correction.get("root_cause") or "").strip()
    kc = (correction.get("kubectl") or "").strip()
    if not rc and not kc:
        return JSONResponse({"error": "Corrección vacía"}, status_code=400)
    text = f"ROOT CAUSE: {rc}\nKUBECTL: {kc}".strip()
    incident_store.update(incident_id, human_correction=text)
    # Captura inmediata: la corrección humana es la señal de mayor calidad.
    record_feedback(incident_store.get(incident_id).to_dict())
    return {"status": "corrected", "id": incident_id}


# ------------------------------------------------------------------
# Demo de remediación (gated por AIOPS_DEMO=true) — inyecta un incidente
# por el camino REAL de remediación sobre un workload desechable, para
# demostrar la aprobación humana y la automática sin esperar una anomalía.
# ------------------------------------------------------------------

@app.get("/api/demo/enabled")
async def demo_enabled():
    """Indica a la consola si debe mostrar los botones de demo."""
    return {"enabled": os.getenv("AIOPS_DEMO", "false").lower() == "true"}


@app.post("/api/demo/incident")
async def demo_incident(mode: str = "human"):
    """Crea un incidente de prueba (L1: rollout restart) sobre el deployment demo.

    mode=human → modo sombra: queda PENDING esperando tu aprobación en /incidents.
    mode=auto  → se ejecuta y verifica automáticamente, sin intervención humana.
    """
    if os.getenv("AIOPS_DEMO", "false").lower() != "true":
        return JSONResponse(
            {"error": "Demo deshabilitado. Arranca el servidor con AIOPS_DEMO=true."},
            status_code=403,
        )
    from types import SimpleNamespace

    from src.remediation.auto_remediation import AutoRemediation
    from src.remediation.base_notifier import build_notifier

    ns = os.getenv("DEMO_NAMESPACE", "aiops-demo")
    deployment = os.getenv("DEMO_DEPLOYMENT", "nginx-demo")
    auto = mode == "auto"

    window = SimpleNamespace(index=999, namespaces={ns}, log_count=12, template_count=3,
                             start_time=0.0, end_time=60.0, raw_logs=["[demo] evento sintético"])
    scored = SimpleNamespace(window=window, score=0.91, model_version=1)
    diagnosis = SimpleNamespace(
        root_cause=(f"[DEMO] El deployment {deployment} en {ns} muestra reinicios "
                    f"elevados; se propone un rollout restart para recuperarlo."),
        kubectl_command=f"kubectl rollout restart deployment/{deployment} -n {ns}",
        react_trace=[],
    )

    rem = AutoRemediation(
        notifier=build_notifier(RemediationConfig()),  # avisa por Teams si está configurado
        max_auto_level=1,
        incident_store=incident_store,      # el mismo que ve la consola web
        shadow_mode=(not auto),             # human → espera aprobación; auto → ejecuta
        verify_wait=int(os.getenv("DEMO_VERIFY_WAIT", "8")),
    )
    rem.handle_async(scored, diagnosis)

    return {
        "mode": mode,
        "namespace": ns,
        "deployment": deployment,
        "kubectl": diagnosis.kubectl_command,
        "hint": ("Aprueba/rechaza en /incidents" if not auto
                 else "Se ejecuta y verifica automáticamente; míralo en /incidents"),
    }


# ------------------------------------------------------------------
# Chat con el cluster — investigación ReAct read-only en vivo (SSE)
# ------------------------------------------------------------------

@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    html_path = Path(__file__).parent / "static" / "chat.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/chat/stream")
async def chat_stream(q: str):
    """Server-Sent Events: emite los pasos del agente conforme investiga."""
    def event_gen():
        if not q.strip():
            yield _sse({"type": "error", "text": "Pregunta vacía"})
            return
        try:
            for event in chat_agent.chat_iter(q):
                yield _sse(event)
        except Exception as e:
            yield _sse({"type": "error", "text": str(e)})
        yield _sse({"type": "done"})

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",  # desactiva buffering en nginx/ingress
    })


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------------
# Topología del cluster — el "cuadro eléctrico"
# ------------------------------------------------------------------

_topology: dict = {"collector": None}


@app.get("/topology", response_class=HTMLResponse)
async def topology_page():
    html_path = Path(__file__).parent / "static" / "topology.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/topology")
async def api_topology():
    try:
        if _topology["collector"] is None:
            _topology["collector"] = TopologyCollector()
        return _topology["collector"].build_graph()
    except Exception as e:
        return JSONResponse({"error": str(e), "nodes": [], "links": [], "stats": {}}, status_code=200)


# ------------------------------------------------------------------
# Seguridad — escaneo de postura read-only
# ------------------------------------------------------------------

_security: dict = {"scanner": None}


@app.get("/security", response_class=HTMLResponse)
async def security_page():
    html_path = Path(__file__).parent / "static" / "security.html"
    return HTMLResponse(html_path.read_text())


@app.get("/api/security")
async def api_security():
    try:
        if _security["scanner"] is None:
            _security["scanner"] = SecurityScanner()
        return _security["scanner"].scan()
    except Exception as e:
        return JSONResponse({"error": str(e), "findings": [], "summary": {}}, status_code=200)


