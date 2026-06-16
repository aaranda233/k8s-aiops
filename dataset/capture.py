"""
Capturador de anomalías reales del pipeline para etiquetado manual.

Cuando el IF detecta una anomalía, guarda los eventos en un fichero
de revisión. Tú los etiquetas con la causa raíz real y quedan
listos para el fine-tuning.

Uso:
  # Ver anomalías pendientes de etiquetar
  python dataset/capture.py list

  # Etiquetar una anomalía
  python dataset/capture.py label <id>

  # Exportar las etiquetadas a JSONL
  python dataset/capture.py export
"""

import json
import sys
from datetime import datetime
from pathlib import Path

CAPTURE_DIR  = Path(__file__).parent / "output" / "captured"
LABELED_FILE = Path(__file__).parent / "output" / "labeled.jsonl"
SYSTEM_PROMPT = (Path(__file__).parent.parent / "src" / "diagnostics" / "ollama_rca.py").read_text().split('"""')[3].strip() if False else \
"""You are an expert Site Reliability Engineer (SRE) specialized in Kubernetes.
You will receive a set of raw Kubernetes events from a time window flagged as anomalous by an ML model.
Your task is to:
1. Identify the root cause of the anomaly in 2-3 sentences.
2. Propose ONE specific kubectl command to investigate or mitigate the issue.

Output format (strict):
ROOT CAUSE: <explanation>
KUBECTL: <exact command>

Be concise. Focus on actionable diagnosis."""


def save_anomaly(window_index: int, score: float, namespaces: list,
                 raw_logs: list, rca_auto: str = "") -> Path:
    """
    Llamado desde el pipeline cuando se detecta una anomalía.
    Guarda los datos para revisión humana posterior.
    """
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = CAPTURE_DIR / f"anomaly_{ts}_W{window_index}.json"

    data = {
        "id": f"{ts}_W{window_index}",
        "captured_at": datetime.now().isoformat(),
        "window_index": window_index,
        "score": score,
        "namespaces": namespaces,
        "raw_logs": raw_logs,
        "rca_auto": rca_auto,   # lo que dijo el SLM zero-shot (referencia)
        "labeled": False,
        "root_cause": "",       # a rellenar manualmente
        "kubectl": "",          # a rellenar manualmente
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return path


def cmd_list() -> None:
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(CAPTURE_DIR.glob("*.json"))
    if not files:
        print("No hay anomalías capturadas.")
        return

    pending   = [f for f in files if not json.loads(f.read_text())["labeled"]]
    labeled   = [f for f in files if json.loads(f.read_text())["labeled"]]

    print(f"\n{'─'*60}")
    print(f"  Anomalías capturadas: {len(files)}  |  Pendientes: {len(pending)}  |  Etiquetadas: {len(labeled)}")
    print(f"{'─'*60}")
    for f in pending[:10]:
        d = json.loads(f.read_text())
        print(f"  [{d['id']}]  score={d['score']:.3f}  ns={d['namespaces']}  logs={len(d['raw_logs'])}")
    if len(pending) > 10:
        print(f"  ... y {len(pending)-10} más")
    print()


def cmd_label(anomaly_id: str) -> None:
    """Interfaz CLI para etiquetar una anomalía."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    matches = list(CAPTURE_DIR.glob(f"*{anomaly_id}*.json"))
    if not matches:
        print(f"No encontrado: {anomaly_id}")
        return

    path = matches[0]
    data = json.loads(path.read_text())

    print(f"\n{'═'*60}")
    print(f"  Anomalía: {data['id']}")
    print(f"  Score: {data['score']:.3f}  |  Namespaces: {data['namespaces']}")
    print(f"{'─'*60}")
    print("  EVENTOS:")
    for log in data["raw_logs"][-20:]:
        print(f"    {log}")
    if data["rca_auto"]:
        print(f"\n  [SLM zero-shot dijo]: {data['rca_auto']}")
    print(f"{'═'*60}\n")

    print("Escribe la causa raíz (Enter para confirmar):")
    root_cause = input("> ").strip()
    if not root_cause:
        print("Cancelado.")
        return

    print("\nEscribe el kubectl recomendado:")
    kubectl = input("> ").strip()
    if not kubectl:
        print("Cancelado.")
        return

    data["root_cause"] = root_cause
    data["kubectl"]    = kubectl
    data["labeled"]    = True
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"\n✓ Etiquetada: {path.name}")


def cmd_export() -> None:
    """Exporta todas las anomalías etiquetadas a JSONL para fine-tuning."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    labeled = [
        json.loads(f.read_text())
        for f in sorted(CAPTURE_DIR.glob("*.json"))
        if json.loads(f.read_text())["labeled"]
    ]

    if not labeled:
        print("No hay anomalías etiquetadas aún.")
        return

    LABELED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELED_FILE, "w") as out:
        for d in labeled:
            sample_logs = d["raw_logs"][-40:]
            user_msg = (
                f"Anomaly Score: {d['score']}\n"
                f"Namespaces: {', '.join(d['namespaces'])}\n"
                f"Total events: {len(d['raw_logs'])}\n"
                f"Event sample:\n" +
                "\n".join(f"  {l}" for l in sample_logs)
            )
            assistant_msg = f"ROOT CAUSE: {d['root_cause']}\nKUBECTL: {d['kubectl']}"
            sample = {
                "messages": [
                    {"role": "system",    "content": SYSTEM_PROMPT},
                    {"role": "user",      "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ],
                "metadata": {
                    "source":    "real_capture",
                    "id":        d["id"],
                    "score":     d["score"],
                    "namespaces": d["namespaces"],
                }
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✓ Exportados {len(labeled)} samples → {LABELED_FILE}")


def cmd_merge() -> None:
    """Combina sintético + capturas reales en un único dataset."""
    synthetic = Path(__file__).parent / "output" / "synthetic.jsonl"
    combined  = Path(__file__).parent / "output" / "combined.jsonl"
    sources   = []

    if synthetic.exists():
        sources.append(synthetic)
        print(f"  + sintético: {synthetic}")
    if LABELED_FILE.exists():
        sources.append(LABELED_FILE)
        print(f"  + real:      {LABELED_FILE}")

    if not sources:
        print("No hay datos. Genera el sintético primero: python dataset/generator.py")
        return

    total = 0
    with open(combined, "w") as out:
        for src in sources:
            for line in src.read_text().splitlines():
                if line.strip():
                    out.write(line + "\n")
                    total += 1

    print(f"\n✓ Dataset combinado: {total} samples → {combined}")


if __name__ == "__main__":
    cmds = {"list": cmd_list, "export": cmd_export, "merge": cmd_merge}
    if len(sys.argv) < 2 or sys.argv[1] not in {*cmds, "label"}:
        print("Uso: python dataset/capture.py [list|label <id>|export|merge]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "label":
        if len(sys.argv) < 3:
            print("Uso: python dataset/capture.py label <id>")
            sys.exit(1)
        cmd_label(sys.argv[2])
    else:
        cmds[cmd]()
