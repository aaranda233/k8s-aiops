"""
k8s-aiops — Pipeline AIOps local para Kubernetes
Punto de entrada.

Uso:
  python main.py replay                        # procesa eventos historicos
  python main.py live                          # stream en vivo
  python main.py replay --namespaces default mcp
  python main.py replay --no-llm
  python main.py replay --threshold 0.75 --bootstrap 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console

from config.settings import (
    CollectorConfig,
    DetectorConfig,
    DiagnosticsConfig,
    PipelineConfig,
)
from src.pipeline import AIOPsPipeline

console = Console()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="k8s-aiops — AIOps pipeline local")
    p.add_argument("mode", choices=["replay", "live"], help="Modo de ejecucion")
    p.add_argument("--namespaces", nargs="+", default=None, help="Namespaces a monitorizar")
    p.add_argument("--no-llm", action="store_true", help="Deshabilitar Capa 3")
    p.add_argument("--threshold", type=float, default=0.80, help="Umbral anomalia [0-1]")
    p.add_argument("--bootstrap", type=int, default=10, help="Ventanas para bootstrap")
    p.add_argument("--window", type=float, default=60.0, help="Tamano ventana (segundos)")
    p.add_argument("--model", type=str, default="qwen2.5-coder:1.5b", help="Modelo Ollama")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    console.print("\n[bold white on blue]  k8s-aiops — Pipeline AIOps Local (CPU-only)  [/]\n")

    cfg = PipelineConfig(
        collector=CollectorConfig(
            namespaces=args.namespaces,
            window_size_seconds=args.window,
            bootstrap_windows=args.bootstrap,
        ),
        detector=DetectorConfig(
            anomaly_threshold=args.threshold,
        ),
        diagnostics=DiagnosticsConfig(
            model=args.model,
            enabled=not args.no_llm,
        ),
        replay_mode=(args.mode == "replay"),
    )

    pipeline = AIOPsPipeline(cfg)

    if args.mode == "replay":
        pipeline.run_replay()
    else:
        pipeline.run_live()


if __name__ == "__main__":
    main()
