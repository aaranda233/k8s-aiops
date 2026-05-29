"""
Servidor FastAPI con WebSocket para la UI en tiempo real.

Arranca el pipeline en un hilo de fondo y emite eventos
a todos los clientes conectados via WebSocket.
"""

import asyncio
import sys
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CollectorConfig, DetectorConfig, DiagnosticsConfig, PipelineConfig
from src.pipeline import AIOPsPipeline
from web.event_bus import EventBus, bus

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
# Pagina principal
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


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
        replay_mode=(mode == "replay"),
    )

    t = threading.Thread(target=_run_pipeline, args=(cfg, mode), daemon=True)
    t.start()
