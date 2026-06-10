"""
Clasificador de riesgo para comandos kubectl.

Level 0 — Solo lectura:      describe, get, logs, top
Level 1 — Reversible local:  rollout restart, scale, rollout undo
Level 2 — Cambio config:     set resources, set image, patch, annotate
Level 3 — Destructivo:       delete, drain, cordon, exec, taint
"""

import shlex
from dataclasses import dataclass

_LEVEL_0_VERBS = {"describe", "get", "logs", "top", "explain", "version", "events"}

_LEVEL_1_PREFIXES = [
    ("rollout", "restart"),
    ("rollout", "undo"),
    ("scale",),
]

_LEVEL_2_PREFIXES = [
    ("set", "resources"),
    ("set", "image"),
    ("set", "env"),
    ("patch",),
    ("annotate",),
    ("label",),
]

_LEVEL_3_VERBS = {
    "delete", "drain", "cordon", "uncordon",
    "taint", "exec", "cp", "replace", "apply",
    "create", "run",
}

LEVEL_LABELS = {
    0: "lectura",
    1: "reversible",
    2: "configuración",
    3: "destructivo",
}


@dataclass(frozen=True)
class RiskResult:
    level: int
    label: str
    reason: str


def score(kubectl_command: str) -> RiskResult:
    """Clasifica un comando kubectl en nivel de riesgo 0-3."""
    cmd = kubectl_command.strip()

    try:
        parts = shlex.split(cmd)
    except ValueError:
        return RiskResult(3, LEVEL_LABELS[3], "No se puede parsear el comando")

    if not parts or parts[0] != "kubectl":
        return RiskResult(3, LEVEL_LABELS[3], "No es un comando kubectl")

    if len(parts) < 2:
        return RiskResult(0, LEVEL_LABELS[0], "Comando vacío")

    verb = parts[1].lower()
    rest = [p.lower() for p in parts[2:]]

    if verb in _LEVEL_0_VERBS:
        return RiskResult(0, LEVEL_LABELS[0], f"Verbo de solo lectura: {verb}")

    if verb in _LEVEL_3_VERBS:
        return RiskResult(3, LEVEL_LABELS[3], f"Verbo destructivo: {verb}")

    all_parts = [verb] + rest
    for prefix in _LEVEL_2_PREFIXES:
        if all_parts[:len(prefix)] == list(prefix):
            return RiskResult(2, LEVEL_LABELS[2], f"Patrón de configuración: {' '.join(prefix)}")

    for prefix in _LEVEL_1_PREFIXES:
        if all_parts[:len(prefix)] == list(prefix):
            return RiskResult(1, LEVEL_LABELS[1], f"Patrón reversible: {' '.join(prefix)}")

    return RiskResult(2, LEVEL_LABELS[2], f"Verbo desconocido '{verb}' — nivel conservador")
