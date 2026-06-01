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


@dataclass
class MLflowConfig:
    enabled: bool = field(default_factory=lambda: os.getenv("MLFLOW_ENABLED", "true").lower() != "false")
    tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://192.168.2.204:30803"))
    experiment: str = field(default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT", "k8s-aiops"))


@dataclass
class PipelineConfig:
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    # Modo replay: procesar eventos historicos en vez de stream en vivo
    replay_mode: bool = False
