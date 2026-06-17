"""Fixtures compartidas para la suite de tests."""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class FakeStep:
    step: int
    thought: str
    action: str | None
    observation: str | None = None
    is_final: bool = False


@dataclass
class FakeWindow:
    index: int = 1
    namespaces: set = field(default_factory=lambda: {"producción"})
    log_count: int = 21
    template_count: int = 5
    start_time: float = 0.0
    end_time: float = 60.0
    raw_logs: list = field(default_factory=lambda: ["evento de ejemplo"])


@dataclass
class FakeScored:
    score: float = 0.91
    model_version: int = 1
    window: FakeWindow = field(default_factory=FakeWindow)


@dataclass
class FakeDiagnosis:
    root_cause: str = "Memory pressure en node-1 causando evictions"
    kubectl_command: str = "kubectl rollout restart deployment/scheduler -n producción"
    react_trace: list = field(default_factory=list)
    namespaces: set = field(default_factory=set)


@pytest.fixture
def scored_window():
    return FakeScored()


@pytest.fixture
def diagnosis():
    return FakeDiagnosis(
        react_trace=[
            FakeStep(1, "Veo evictions repetidas", "kubectl describe node node-1 -n producción"),
            FakeStep(2, "Nodo al 97% de memoria", None, is_final=True),
        ]
    )
