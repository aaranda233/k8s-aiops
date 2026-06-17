"""
Recuperador de incidentes pasados (RAG) — aprendizaje SIN reentrenar.

Indexa los incidentes resueltos/validados (de feedback.jsonl) y, ante un
incidente nuevo, recupera los casos pasados más parecidos para inyectarlos como
contexto en el prompt del RCA. El modelo mejora por el CONTEXTO, no por los
pesos: el conocimiento nuevo está disponible al instante, sin GPU.

Recuperación léxica TF-IDF (sklearn — ya es dependencia; cero deps nuevas y
corre en CPU). Para un 1.5B con num_ctx=2048 el contexto es oro: se inyectan
muy pocos casos (1-2), resumidos a causa+fix.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class IncidentRetriever:
    def __init__(self, cases: list[dict]):
        # cases: [{"text": <eventos>, "root_cause": ..., "kubectl": ...}]
        self.cases = cases
        self._vec = None
        self._matrix = None
        texts = [c.get("text", "") for c in cases if c.get("text")]
        if texts:
            try:
                self._vec = TfidfVectorizer(min_df=1, stop_words=None)
                self._matrix = self._vec.fit_transform(texts)
            except ValueError:
                # vocabulario vacío (p.ej. solo stopwords) → sin índice
                self._vec = None

    def retrieve(self, query: str, k: int = 2, min_score: float = 0.05) -> list[dict]:
        """Top-k casos pasados más similares al query (eventos del incidente nuevo)."""
        if not self._vec or not self.cases or not query.strip():
            return []
        qv = self._vec.transform([query])
        sims = cosine_similarity(qv, self._matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [
            {**self.cases[i], "score": round(float(sims[i]), 3)}
            for i in order if sims[i] >= min_score
        ]

    @classmethod
    def from_sources(cls, feedback_path: str, corpus_path: str | None = None,
                     positive_only: bool = True) -> "IncidentRetriever":
        """Índice combinado: memoria del bucle (feedback) + corpus de casos conocidos.

        Permite que RAG funcione desde el primer día (corpus) y mejore con el
        feedback real acumulado.
        """
        cases = cls.from_feedback(feedback_path, positive_only).cases
        if corpus_path:
            cases = cases + _load_corpus_cases(corpus_path)
        return cls(cases)

    @classmethod
    def from_feedback(cls, path: str, positive_only: bool = True) -> "IncidentRetriever":
        """Construye el índice desde feedback.jsonl (memoria de casos validados)."""
        cases: list[dict] = []
        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if positive_only and ex.get("label") != "positive" and not ex.get("human_correction"):
                        continue
                    # preferir la corrección humana como "verdad" del caso
                    rc = ex.get("root_cause", "")
                    kc = ex.get("kubectl_cmd", "")
                    corr = ex.get("human_correction") or ""
                    if corr:
                        rc, kc = _split_correction(corr, rc, kc)
                    cases.append({
                        "text": (ex.get("prompt", {}) or {}).get("user", ""),
                        "root_cause": rc,
                        "kubectl": kc,
                        "namespaces": ex.get("namespaces", []),
                    })
        return cls(cases)


def _load_corpus_cases(path: str) -> list[dict]:
    """Carga casos pasados de un dataset (formato SFT messages o preferencia)."""
    from src.diagnostics.ollama_rca import parse_diagnosis

    cases: list[dict] = []
    p = Path(path)
    if not p.exists():
        return cases
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            user = answer = None
            msgs = ex.get("messages")
            if msgs and len(msgs) >= 3:
                user, answer = msgs[1]["content"], msgs[2]["content"]
            elif ex.get("prompt") and ex.get("chosen"):
                user, answer = ex["prompt"][1]["content"], ex["chosen"][0]["content"]
            if not user or not answer:
                continue
            rc, kc = parse_diagnosis(answer)
            cases.append({"text": user, "root_cause": rc, "kubectl": kc, "namespaces": []})
    return cases


def _split_correction(correction: str, default_rc: str, default_kc: str) -> tuple[str, str]:
    rc, kc = default_rc, default_kc
    for line in correction.splitlines():
        s = line.strip()
        if s.upper().startswith("ROOT CAUSE:"):
            rc = s.split(":", 1)[1].strip()
        elif s.upper().startswith("KUBECTL:"):
            kc = s.split(":", 1)[1].strip()
    return rc, kc


def rag_context(cases: list[dict], max_chars: int = 600) -> str:
    """Formatea los casos recuperados como bloque de contexto acotado para el prompt."""
    if not cases:
        return ""
    lines = ["Similar past incidents (already resolved on this cluster):"]
    for c in cases:
        rc = (c.get("root_cause", "") or "")[:200]
        kc = (c.get("kubectl", "") or "")[:120]
        lines.append(f"- [sim {c.get('score', 0)}] CAUSE: {rc} | FIX: {kc}")
    text = "\n".join(lines)
    return text[:max_chars]
