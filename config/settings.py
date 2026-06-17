"""Configuracion centralizada del pipeline."""

import os
from dataclasses import dataclass, field


@dataclass
class CollectorConfig:
    # Namespaces a monitorizar (None = todos)
    namespaces: list[str] | None = None
    # Tamano de ventana temporal en segundos
    window_size_seconds: float = 60.0
    # Cuantas ventanas acumular antes de arrancar deteccion (fase bootstrap)
    bootstrap_windows: int = 10
    # Cuantas ventanas recientes mantener para reentrenamiento (ventana deslizante)
    rolling_window_size: int = 50
    # Reentrenar IF cada N ventanas nuevas
    retrain_every_n_windows: int = 5


@dataclass
class LogConfig:
    # Detección sobre logs de aplicación (además de eventos K8s). Opt-in.
    enabled: bool = field(default_factory=lambda: os.getenv("LOG_COLLECTION_ENABLED", "false").lower() == "true")
    # Namespaces a leer (coma-separados). OBLIGATORIO si enabled — nunca todo el cluster.
    namespaces: list[str] = field(default_factory=lambda: [
        ns for ns in os.getenv("LOG_NAMESPACES", "").split(",") if ns.strip()
    ])
    poll_interval: float = field(default_factory=lambda: float(os.getenv("LOG_POLL_INTERVAL", "30")))
    tail_lines: int = field(default_factory=lambda: int(os.getenv("LOG_TAIL_LINES", "50")))
    since_seconds: int = field(default_factory=lambda: int(os.getenv("LOG_SINCE_SECONDS", "35")))
    max_pods: int = field(default_factory=lambda: int(os.getenv("LOG_MAX_PODS", "50")))


@dataclass
class DetectorConfig:
    # Score minimo para disparar alerta (0-1)
    anomaly_threshold: float = 0.80
    # Parametros de Isolation Forest
    n_estimators: int = 200
    contamination: float = 0.05
    random_state: int = 42


@dataclass
class DiagnosticsConfig:
    host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b"))
    enabled: bool = True
    max_logs_in_prompt: int = 40
    timeout_seconds: float = 120.0
    # Modo de diagnóstico: "single_shot" | "react" | "hybrid"
    #   single_shot — OllamaRCA, una llamada al fine-tuneado (default)
    #   react       — ReActAgent, fine-tuneado intenta formato ReAct
    #   hybrid      — HybridReActAgent: base investiga + fine-tuneado diagnostica
    react_mode: str = field(default_factory=lambda: os.getenv("REACT_MODE", "single_shot"))
    react_base_model: str = field(default_factory=lambda: os.getenv("REACT_BASE_MODEL", "qwen2.5:1.5b"))
    react_max_steps: int = 3
    react_dry_run: bool = field(default_factory=lambda: os.getenv("REACT_DRY_RUN", "true").lower() != "false")


@dataclass
class RemediationConfig:
    enabled: bool = field(default_factory=lambda: os.getenv("REMEDIATION_ENABLED", "false").lower() == "true")
    # Nivel máximo de acción automática: 1=restart/scale, 2=patch/config, 3=nunca auto
    max_auto_level: int = field(default_factory=lambda: int(os.getenv("REMEDIATION_MAX_LEVEL", "1")))
    # Modo sombra: genera incidentes y propone fixes pero NO auto-ejecuta nada;
    # todo (incluido Level 1) pasa por aprobación humana en la consola.
    shadow_mode: bool = field(default_factory=lambda: os.getenv("REMEDIATION_SHADOW", "false").lower() == "true")
    # Segundos de espera tras el fix para verificar
    verify_wait_seconds: int = 90
    # Ventana de deduplicación: un mismo problema (mismos namespaces) que recurre
    # dentro de este tiempo no crea un incidente nuevo, sino que incrementa el
    # contador del existente. 0 = sin deduplicación.
    dedup_window_seconds: int = field(default_factory=lambda: int(os.getenv("REMEDIATION_DEDUP_WINDOW", "1800")))
    # Canal de notificación: teams | email | both | none
    notify_channel: str = field(default_factory=lambda: os.getenv("NOTIFY_CHANNEL", "teams"))
    # Teams — Incoming Webhook del canal de ops
    teams_webhook_url: str = field(default_factory=lambda: os.getenv("TEAMS_WEBHOOK_URL", ""))
    # SMTP (canal email / fallback)
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", "smtp.gmail.com"))
    smtp_port: int = field(default_factory=lambda: int(os.getenv("SMTP_PORT", "587")))
    smtp_user: str = field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    smtp_pass: str = field(default_factory=lambda: os.getenv("SMTP_PASS", ""))
    smtp_from: str = field(default_factory=lambda: os.getenv("SMTP_FROM", ""))
    notify_email: str = field(default_factory=lambda: os.getenv("NOTIFY_EMAIL", ""))
    # URL base pública para los links de aprobación en las notificaciones
    webhook_base_url: str = field(default_factory=lambda: os.getenv("WEBHOOK_BASE_URL", "http://localhost:8000"))


@dataclass
class MLflowConfig:
    enabled: bool = field(default_factory=lambda: os.getenv("MLFLOW_ENABLED", "true").lower() != "false")
    tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://192.168.2.204:30803"))
    experiment: str = field(default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT", "k8s-aiops"))


@dataclass
class PipelineConfig:
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    logs: LogConfig = field(default_factory=LogConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    remediation: RemediationConfig = field(default_factory=RemediationConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    # Modo replay: procesar eventos historicos en vez de stream en vivo
    replay_mode: bool = False
