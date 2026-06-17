"""
Comparativa de vías de aprendizaje: RAG (recuperación, sin reentrenar) frente al
modelo plain, sobre el mismo test set y las mismas métricas (Parse%/Keyword%).

La parte de fine-tuning (continual ORPO) se compara con los números de los
experimentos ya registrados (RESEARCH.md) + las versiones que produzca el bucle
con GPU; aquí se mide empíricamente lo que corre en CPU: el efecto del RAG.

La lógica de evaluación (evaluate_samples / compare) es pura y testeable con un
modelo simulado; main() la conecta con Ollama y un corpus de casos pasados.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import keyword_hit, parse_rate
from src.diagnostics.incident_retriever import IncidentRetriever, rag_context
from src.diagnostics.ollama_rca import parse_diagnosis


def evaluate_samples(samples: list[dict], call_fn, retriever: IncidentRetriever | None = None,
                     rag_k: int = 2) -> dict:
    """Evalúa el test set con un modelo (call_fn(system,user)->texto).

    Si se pasa retriever, antepone contexto RAG al user prompt. Devuelve métricas
    agregadas: parse_rate, keyword_hit, latency_mean.
    """
    parsed_n = kw_n = 0
    latencies = []
    for s in samples:
        msgs = s["messages"]
        system = msgs[0]["content"]
        user = msgs[1]["content"]
        scenario = s.get("metadata", {}).get("scenario_id", "")

        if retriever is not None:
            ctx = rag_context(retriever.retrieve(user, k=rag_k))
            if ctx:
                user = f"{ctx}\n\n{user}"

        t0 = time.time()
        text = call_fn(system, user)
        latencies.append(time.time() - t0)

        rc, kc = parse_diagnosis(text)
        if parse_rate(rc, kc):
            parsed_n += 1
        if keyword_hit(rc, scenario):
            kw_n += 1

    n = len(samples) or 1
    return {
        "n": len(samples),
        "parse_rate": round(parsed_n / n, 3),
        "keyword_hit": round(kw_n / n, 3),
        "latency_mean": round(sum(latencies) / n, 3) if latencies else 0.0,
    }


def compare(samples: list[dict], retriever: IncidentRetriever, call_fn) -> dict:
    """Compara plain vs RAG con el mismo modelo y test set."""
    plain = evaluate_samples(samples, call_fn, retriever=None)
    rag = evaluate_samples(samples, call_fn, retriever=retriever)
    return {"plain": plain, "rag": rag}


# ── Carga de datos ──────────────────────────────────────────────────────────

def load_test_set(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def corpus_from_messages(path: str) -> IncidentRetriever:
    """Construye un corpus de recuperación desde un jsonl de casos pasados.

    Acepta dos formatos: SFT (messages=[system,user,assistant]) y preferencia
    (prompt=[system,user], chosen=[assistant]).
    """
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            user = answer = None
            msgs = ex.get("messages")
            if msgs and len(msgs) >= 3:
                user, answer = msgs[1]["content"], msgs[2]["content"]
            elif ex.get("prompt") and ex.get("chosen"):
                user = ex["prompt"][1]["content"]
                answer = ex["chosen"][0]["content"]
            if not user or not answer:
                continue
            rc, kc = parse_diagnosis(answer)
            cases.append({"text": user, "root_cause": rc, "kubectl": kc})
    return IncidentRetriever(cases)


# ── Ejecución en vivo (Ollama) ──────────────────────────────────────────────

def _ollama_caller(model: str, host: str):
    import httpx

    def call(system: str, user: str) -> str:
        payload = {"model": model, "stream": False,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "options": {"temperature": 0.1, "num_predict": 300, "num_ctx": 2048}}
        try:
            r = httpx.post(f"{host}/api/chat", json=payload, timeout=120)
            return r.json()["message"]["content"]
        except Exception as e:
            return f"(error: {e})"
    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="k8s-rca-orpo:latest")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--test-set", default="eval/test_set.jsonl")
    ap.add_argument("--corpus", default="dataset/output/dpo_dataset_v2.jsonl",
                    help="casos pasados para el índice RAG")
    args = ap.parse_args()

    samples = load_test_set(args.test_set)
    retriever = corpus_from_messages(args.corpus)
    print(f"test set: {len(samples)} | corpus RAG: {len(retriever.cases)} casos")

    result = compare(samples, retriever, _ollama_caller(args.model, args.host))
    print("\n=== COMPARATIVA (mismo modelo, mismo test set) ===")
    print(f"{'métrica':<16}{'plain':>10}{'RAG':>10}{'Δ':>10}")
    for m in ("parse_rate", "keyword_hit", "latency_mean"):
        p, r = result["plain"][m], result["rag"][m]
        print(f"{m:<16}{p:>10.3f}{r:>10.3f}{r - p:>+10.3f}")
    Path("eval/results").mkdir(parents=True, exist_ok=True)
    Path("eval/results/compare_rag.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
