"""
Captura de feedback del bucle cerrado.

Cada incidente que alcanza un estado terminal (con decisión humana y/o
resultado de verificación) se convierte en un ejemplo de entrenamiento y se
añade a feedback.jsonl. Es la materia prima del reentrenamiento de preferencias
(ORPO): approved+resolved -> señal positiva; rejected/failed -> señal negativa.

Se engancha como callback en IncidentStore.set_feedback_hook().
"""

import json
import os
import time
from pathlib import Path

from src.diagnostics.ollama_rca import _SYSTEM_PROMPT

POSITIVE = "positive"
NEGATIVE = "negative"
AMBIGUOUS = "ambiguous"

_DEFAULT_PATH = "data/feedback/feedback.jsonl"


def feedback_path() -> str:
    return os.getenv("AIOPS_FEEDBACK_FILE", _DEFAULT_PATH)


def derive_label(response, status, verified) -> str:
    """Deriva la etiqueta de aprendizaje del resultado del incidente.

    positive  : el diagnóstico/fix fue validado (humano aprobó y/o se verificó).
    negative  : rechazado por el humano o el fix falló (mala recomendación).
    ambiguous : sin señal clara (timeout, escalado, L0 auto sin humano) -> se excluye.
    """
    if status == "resolved" and (response == "approved" or verified is True):
        return POSITIVE
    if response == "rejected" or status == "failed":
        return NEGATIVE
    return AMBIGUOUS


def build_example(incident: dict) -> dict | None:
    """Construye el ejemplo de feedback desde el dict del incidente."""
    prompt_user = incident.get("prompt_user", "")
    if not prompt_user:
        return None  # sin el input exacto del modelo no hay ejemplo fiel
    label = derive_label(
        incident.get("response"), incident.get("status"), incident.get("verified")
    )
    correction = incident.get("human_correction", "") or None
    return {
        "incident_id": incident.get("id"),
        "captured_at": time.time(),
        "prompt": {"system": _SYSTEM_PROMPT, "user": prompt_user},
        "model_output": (
            f"ROOT CAUSE: {incident.get('root_cause', '')}\n"
            f"KUBECTL: {incident.get('kubectl_cmd', '')}"
        ),
        "root_cause": incident.get("root_cause", ""),
        "kubectl_cmd": incident.get("kubectl_cmd", ""),
        "response": incident.get("response"),
        "status": incident.get("status"),
        "verified": incident.get("verified"),
        "risk_level": incident.get("risk_level"),
        "namespaces": incident.get("namespaces", []),
        "score": incident.get("score"),
        "label": label,
        "human_correction": correction,
        "source": "closed_loop",
        # Procedencia del grafo (para consolidación verificada y atribución).
        "solution_source": incident.get("solution_source", "catalog"),
        "solution_key": incident.get("solution_key", ""),
    }


def record_feedback(incident: dict, path: str | None = None) -> dict | None:
    """Callback de IncidentStore: añade un ejemplo a feedback.jsonl. No lanza."""
    example = build_example(incident)
    if example is None:
        return None
    p = Path(path or feedback_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")
    return example
