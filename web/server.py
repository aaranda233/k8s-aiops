"""
Servidor FastAPI con WebSocket para la UI en tiempo real.

Arranca el pipeline en un hilo de fondo y emite eventos
a todos los clientes conectados via WebSocket.
"""

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CollectorConfig, DetectorConfig, DiagnosticsConfig, PipelineConfig, RemediationConfig
from src.diagnostics.cluster_chat import ClusterChatAgent
from src.pipeline import AIOPsPipeline
from src.remediation.incident_store import IncidentStore
from web.event_bus import bus

# Registro de incidentes compartido entre el pipeline (remediación) y la consola
incident_store = IncidentStore()

# Agente de chat read-only (investigación on-demand del cluster)
chat_agent = ClusterChatAgent(
    host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    model=os.getenv("REACT_BASE_MODEL", "qwen2.5:1.5b"),       # investigador
    expert_model=os.getenv("OLLAMA_MODEL", "k8s-rca-orpo:latest"),  # sintetiza la conclusión
    max_steps=int(os.getenv("CHAT_MAX_STEPS", "5")),
    dry_run=os.getenv("CHAT_DRY_RUN", "false").lower() == "true",
)

app = FastAPI(title="k8s-aiops")

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
# Estado de la aplicación (para sondas de K8s)
# ------------------------------------------------------------------

_app_state = {"ready": False, "pipeline_thread": None}


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
# Arranque del pipeline en hilo de fondo
# ------------------------------------------------------------------

def _run_pipeline(cfg: PipelineConfig, mode: str) -> None:
    pipeline = AIOPsPipeline(cfg=cfg, event_bus=bus)
    if mode == "replay":
        pipeline.run_replay()
    else:
        pipeline.run_live()


@app.on_event("startup")
async def startup():
    # Registrar el event loop del servidor en el bus
    loop = asyncio.get_event_loop()
    bus.set_loop(loop)

    # Configuracion del pipeline — ajusta aqui o via variables de entorno
    import os
    mode = os.getenv("PIPELINE_MODE", "live")
    bootstrap = int(os.getenv("BOOTSTRAP_WINDOWS", "5"))
    threshold = float(os.getenv("ANOMALY_THRESHOLD", "0.80"))
    window_size = float(os.getenv("WINDOW_SIZE", "60"))
    namespaces_env = os.getenv("NAMESPACES", "")
    namespaces = namespaces_env.split(",") if namespaces_env else None

    cfg = PipelineConfig(
        collector=CollectorConfig(
            namespaces=namespaces,
            window_size_seconds=window_size,
            bootstrap_windows=bootstrap,
        ),
        detector=DetectorConfig(anomaly_threshold=threshold),
        diagnostics=DiagnosticsConfig(enabled=False),  # LLM se activa manualmente
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
            import logging
            logging.getLogger(__name__).error("Pipeline no pudo arrancar: %s", e)
            _app_state["ready"] = False

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    _app_state["pipeline_thread"] = t
