"""
Tests del grounding semántico (src/diagnostics/semantic_ground.py).

Embeddings mockeados (vectores deterministas) para validar la lógica de
selección: umbral, margen sobre el segundo, filtro por namespaces permitidos y
degradación a None cuando no hay embeddings.
"""

import pytest

from src.diagnostics import semantic_ground as sg
from src.diagnostics.semantic_ground import NamespaceGrounder, _cosine, _image_base


@pytest.mark.unit
def test_image_base():
    assert _image_base("docker.io/library/postgres:15") == "postgres"
    assert _image_base("nginx:alpine") == "nginx"
    assert _image_base("curlimages/curl@sha256:abc") == "curl"


@pytest.mark.unit
def test_cosine_basic():
    assert _cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)


def _grounder_with(embeddings: dict[str, list[float]]) -> NamespaceGrounder:
    g = NamespaceGrounder()
    g._emb = embeddings
    g._built = True
    return g


@pytest.mark.unit
def test_ground_picks_nearest_above_threshold(monkeypatch):
    # "base de datos" ≈ vector de postgresql; lejos de los demás
    g = _grounder_with({
        "postgresql": [1.0, 0.0, 0.0],
        "argocd": [0.0, 1.0, 0.0],
        "llm-app": [0.0, 0.0, 1.0],
    })
    monkeypatch.setattr(sg, "_embed", lambda t: [0.95, 0.1, 0.0])
    assert g.ground("base de datos", {"postgresql", "argocd", "llm-app"}) == "postgresql"


@pytest.mark.unit
def test_ground_returns_none_when_ambiguous(monkeypatch):
    # Dos candidatos casi empatados → margen insuficiente → None (no arriesga)
    g = _grounder_with({"a": [1.0, 0.0], "b": [0.98, 0.02]})
    monkeypatch.setattr(sg, "_embed", lambda t: [1.0, 0.0])
    assert g.ground("algo", {"a", "b"}) is None


@pytest.mark.unit
def test_ground_returns_none_below_threshold(monkeypatch):
    g = _grounder_with({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    monkeypatch.setattr(sg, "_embed", lambda t: [0.5, 0.5])  # coseno ~0.707 vs 0.707 → empate
    # margen 0 → None
    assert g.ground("algo", {"a", "b"}) is None


@pytest.mark.unit
def test_ground_filters_allowed(monkeypatch):
    g = _grounder_with({"postgresql": [1.0, 0.0], "argocd": [0.0, 1.0]})
    monkeypatch.setattr(sg, "_embed", lambda t: [1.0, 0.0])  # query ≈ postgresql
    # postgresql sería el mejor, pero no está en allowed → solo argocd (coseno 0) → None
    assert g.ground("base de datos", {"argocd"}) is None


@pytest.mark.unit
def test_ground_none_when_embed_unavailable(monkeypatch):
    g = _grounder_with({"postgresql": [1.0, 0.0]})
    monkeypatch.setattr(sg, "_embed", lambda t: None)  # embeddings caídos
    assert g.ground("base de datos", {"postgresql"}) is None
