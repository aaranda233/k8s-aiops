"""
Versionado y despliegue del experto RCA en Ollama, con registro y rollback.

Cada modelo entrenado se registra como k8s-rca-orpo-v{N}; un alias estable
k8s-rca-orpo apunta a la versión activa. El registro (model_registry.json)
guarda versión, métricas del gate y el dataset que la entrenó (trazabilidad).
Rollback = repuntar el alias a la versión anterior.

La lógica del registro es pura/testeable; create/alias en Ollama son subprocess.
"""

import json
import subprocess
import time
from pathlib import Path

REGISTRY_PATH = "finetune/output/model_registry.json"
ALIAS = "k8s-rca-orpo"


class ModelRegistry:
    def __init__(self, path: str = REGISTRY_PATH):
        self.path = Path(path)
        self._data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                pass
        return {"active": None, "versions": []}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))

    def next_version(self) -> int:
        return len(self._data["versions"]) + 1

    def record(self, version: int, gguf_path: str, gate: dict, dataset: str,
               promoted: bool, feedback_count: int = 0) -> None:
        self._data["versions"].append({
            "version": version,
            "model": f"{ALIAS}-v{version}",
            "gguf": gguf_path,
            "dataset": dataset,
            "gate": gate,
            "promoted": promoted,
            "feedback_count": feedback_count,  # nº ejemplos consolidados en esta versión
            "created_at": time.time(),
        })
        if promoted:
            self._data["active"] = version
        self._save()

    def consolidation_watermark(self) -> int:
        """Nº de ejemplos de feedback ya consolidados en el modelo ACTIVO.

        El RAG excluye los ejemplos hasta este punto (ya están en los pesos). Si
        no hay versión activa, 0 (RAG retiene todo el feedback). En rollback, el
        watermark sigue al modelo activo automáticamente.
        """
        active = self._data["active"]
        if active is None:
            return 0
        for v in self._data["versions"]:
            if v["version"] == active:
                return v.get("feedback_count", 0)
        return 0

    def active(self) -> int | None:
        return self._data["active"]

    def active_model(self) -> str | None:
        v = self._data["active"]
        return f"{ALIAS}-v{v}" if v else None

    def history(self) -> list[dict]:
        return list(self._data["versions"])

    def rollback(self) -> int | None:
        """Vuelve a la versión promocionada anterior. Devuelve la nueva activa."""
        promoted = [v["version"] for v in self._data["versions"] if v["promoted"]]
        if len(promoted) < 2:
            return self._data["active"]
        # penúltima promocionada
        self._data["active"] = promoted[-2]
        self._save()
        return self._data["active"]


def ollama_create(version: int, gguf_path: str, modelfile: str = "finetune/Modelfile_orpo") -> bool:
    """Crea k8s-rca-orpo-v{N} en Ollama desde el GGUF. Devuelve éxito."""
    model = f"{ALIAS}-v{version}"
    try:
        r = subprocess.run(["ollama", "create", model, "-f", modelfile],
                           capture_output=True, text=True, timeout=300)
        return r.returncode == 0
    except Exception:
        return False


def set_alias(version: int) -> bool:
    """Apunta el alias estable k8s-rca-orpo a la versión dada (copy en Ollama)."""
    try:
        r = subprocess.run(["ollama", "cp", f"{ALIAS}-v{version}", ALIAS],
                           capture_output=True, text=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False
