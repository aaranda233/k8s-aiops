"""
Regresión: la serialización del trace del RCA debe soportar AMBOS tipos de paso.

El pipeline emite el trace a la web accediendo a un flag de "paso final". El modo
react usa TraceStep.is_final; el modo hybrid usa InvestigationStep.is_done. Acceder
a is_final sin fallback rompía el RCA en modo hybrid (AttributeError), dejando la
consola de incidencias vacía pese a haber una anomalía real. Ver pipeline._trigger_rca.
"""

import pytest

from src.diagnostics.hybrid_react_agent import InvestigationStep
from src.diagnostics.react_agent import TraceStep


def _final_flag(step) -> bool:
    # Misma expresión que usa el pipeline al construir el evento "rca".
    return getattr(step, "is_final", getattr(step, "is_done", False))


@pytest.mark.unit
def test_hybrid_investigation_step_has_no_is_final():
    """Garantiza la premisa del bug: InvestigationStep NO tiene is_final."""
    step = InvestigationStep(step=1, thought="t", action="kubectl get pods",
                             observation="ok", is_done=True)
    assert not hasattr(step, "is_final")
    assert _final_flag(step) is True


@pytest.mark.unit
def test_react_trace_step_has_no_is_done():
    step = TraceStep(step=1, thought="t", action=None, observation=None, is_final=True)
    assert not hasattr(step, "is_done")
    assert _final_flag(step) is True


@pytest.mark.unit
def test_final_flag_defaults_false_when_not_final():
    assert _final_flag(InvestigationStep(step=1, thought="t", action="x", observation=None)) is False
    assert _final_flag(TraceStep(step=1, thought="t", action="x", observation=None)) is False
