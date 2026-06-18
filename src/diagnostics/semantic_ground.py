"""
Grounding semántico de la pregunta → namespace.

Cuando el operador no escribe el nombre EXACTO del namespace ("¿qué le pasa a la
base de datos?" en vez de "postgresql"), mapeamos la pregunta al namespace por
SIMILITUD de significado (embeddings), no por coincidencia literal.

Cómo:
  1. Cada namespace se describe por el TIPO de lo que ejecuta, derivado de las
     imágenes de sus pods (postgresql → "postgres", aiops-demo → "nginx").
  2. Se embeben esas descripciones y la pregunta (nomic-embed-text vía Ollama).
  3. ground() devuelve el namespace más cercano por coseno SI supera un umbral y
     gana al segundo por un margen (para no arriesgar groundings ambiguos).

Determinista en el resultado (dado el modelo de embeddings); degrada a None si
los embeddings no están disponibles → el chat cae al matching exacto. El modelo
NO decide el control: solo se usa para recuperar, no para generar.
"""

import json
import math
import os
import subprocess

import httpx

_EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.45"))
_MARGIN = float(os.getenv("GROUNDING_MARGIN", "0.03"))
_ENABLED = os.getenv("GROUNDING_ENABLED", "true").lower() != "false"


def _embed(text: str) -> list[float] | None:
    try:
        resp = httpx.post(
            f"{_OLLAMA_HOST}/api/embeddings",
            json={"model": _EMBED_MODEL, "prompt": text},
            timeout=15,
        )
        resp.raise_for_status()
        emb = resp.json().get("embedding")
        return emb if emb else None
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    return num / (da * db) if da and db else 0.0


def _image_base(image: str) -> str:
    """'docker.io/library/postgres:15' → 'postgres'."""
    return image.split("@", 1)[0].split(":", 1)[0].rsplit("/", 1)[-1]


class NamespaceGrounder:
    """Resuelve la pregunta a un namespace por similitud semántica (con caché)."""

    def __init__(self):
        self._emb: dict[str, list[float]] = {}
        self._built = False

    def _namespace_descriptions(self) -> dict[str, str]:
        """{ns: 'namespace <ns>: <imágenes>'} derivado de los pods del cluster."""
        try:
            proc = subprocess.run(
                ["kubectl", "get", "pods", "-A", "-o", "json"],
                capture_output=True, text=True, timeout=20,
            )
            data = json.loads(proc.stdout)
        except Exception:
            return {}
        agg: dict[str, set[str]] = {}
        for item in data.get("items", []):
            ns = item.get("metadata", {}).get("namespace", "")
            if not ns:
                continue
            toks = agg.setdefault(ns, set())
            for c in item.get("spec", {}).get("containers", []):
                img = c.get("image", "")
                if img:
                    toks.add(_image_base(img))
        return {ns: f"namespace {ns}: {' '.join(sorted(t))}" for ns, t in agg.items() if t}

    def _build(self) -> None:
        self._built = True
        for ns, desc in self._namespace_descriptions().items():
            emb = _embed("search_document: " + desc)
            if emb:
                self._emb[ns] = emb

    def reset(self) -> None:
        """Fuerza reconstruir la caché (p. ej. tras despliegues nuevos)."""
        self._emb = {}
        self._built = False

    def ground(self, question: str, allowed: set[str]) -> str | None:
        """Namespace al que se refiere la pregunta, o None si no hay confianza."""
        if not _ENABLED or not question or not allowed:
            return None
        if not self._built:
            self._build()
        cands = [ns for ns in self._emb if ns in allowed]
        if not cands:
            return None
        q_emb = _embed("search_query: " + question)
        if not q_emb:
            return None
        scored = sorted(((_cosine(q_emb, self._emb[ns]), ns) for ns in cands), reverse=True)
        best_score, best_ns = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= _THRESHOLD and (best_score - second) >= _MARGIN:
            return best_ns
        return None
