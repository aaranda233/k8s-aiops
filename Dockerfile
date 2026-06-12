# ──────────────────────────────────────────────────────────────────
# k8s-aiops — imagen de la aplicación (pipeline + web UI)
# CPU-only. El modelo se sirve aparte vía Ollama.
# ──────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# kubectl — necesario para la capa de investigación y remediación
ARG KUBECTL_VERSION=v1.30.0
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/amd64/kubectl" -o /usr/local/bin/kubectl \
    && chmod +x /usr/local/bin/kubectl \
    && apt-get purge -y curl && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencias primero (mejor cache de capas)
# drain3 0.9.11 fija cachetools==4.2.1, pero mlflow requiere >=5.0.0.
# drain3 funciona con cachetools 5.x en runtime (API LRUCache sin cambios),
# así que lo instalamos sin sus pins rígidos y verificamos que importe.
COPY requirements.txt .
RUN pip install --no-cache-dir "cachetools>=5.0.0,<6" "jsonpickle>=2.0" \
    && pip install --no-cache-dir --no-deps drain3==0.9.11 \
    && grep -v '^drain3' requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt \
    && python -c "from drain3 import TemplateMiner; import cachetools; print('drain3 OK con cachetools', cachetools.__version__)"

# Código de la aplicación
COPY config/ ./config/
COPY src/ ./src/
COPY web/ ./web/
COPY dataset/ ./dataset/
COPY main.py .

# Usuario no-root
RUN useradd --create-home --uid 1000 aiops && chown -R aiops:aiops /app
USER aiops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=4).raise_for_status()" || exit 1

CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8000"]
