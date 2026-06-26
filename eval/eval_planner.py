#!/usr/bin/env python3
"""Evaluación del planner agéntico vs single-shot (E4).

Para cada clase de fallo: inyecta el fallo en un namespace vivo (así el planner
puede investigar recursos reales), y compara, con el MISMO modelo grande
(`qwen2.5-coder:14b`), dos modos de escalado:
  - single-shot: una llamada ciega → plan JSON.
  - agentic:     investiga el cluster en read-only y luego propone el plan.
Mide: % de planes con acción ejecutable, % sin placeholders `<...>`, % seguros
(todos los comandos pasan el validador), y nº de pasos. Pensado para el SERVIDOR.

Uso:  python eval/eval_planner.py --faults oom,image,crashloop --repeat 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.chaos_runner import FAULTS, cleanup, ensure_ns, inject, kubectl  # noqa: E402

_PLACEHOLDER = re.compile(r"<[^>]+>")

# rc corto por clase (lo que daría la capa de diagnóstico)
_RC = {
    "crashloop": "El deployment entra en CrashLoopBackOff por un fallo de arranque.",
    "oom": "El contenedor es OOMKilled: el límite de memoria es insuficiente.",
    "image": "ImagePullBackOff: no se puede descargar la imagen (tag/registry).",
}


def wait_pod(ns: str, timeout: float = 60.0) -> None:
    """Espera a que haya algún pod (aunque falle) en el namespace."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        out, _ = kubectl(["get", "pods", "-n", ns, "--no-headers"])
        if out.strip() and "No resources" not in out:
            return
        time.sleep(4)


def live_evidence(ns: str) -> str:
    """Evidencia read-only real del namespace (eventos) para alimentar el escalado."""
    out, _ = kubectl(["get", "events", "-n", ns, "--sort-by=.lastTimestamp"])
    return f"Namespace {ns}\nEventos:\n{out[:1500]}"


def score_plan(steps) -> dict:
    from src.diagnostics import escalation
    n = len(steps)
    has_cmd = any(getattr(s, "action_type", "") == "command" for s in steps)
    no_ph = all(not _PLACEHOLDER.search(s.action) for s in steps) if steps else False
    safe = all(escalation._command_is_safe(s.action) for s in steps) if steps else False
    return {"steps": n, "executable": has_cmd, "no_placeholder": no_ph, "safe": safe}


def run_one(fault: str, base: str, idx: int) -> dict:
    from src.diagnostics import escalation
    ns = f"{base}-{fault}-{idx}"
    ensure_ns(ns)
    inject(fault, ns, f"plan-{fault}")
    wait_pod(ns)
    time.sleep(8)  # deja que se generen eventos
    rc = _RC[fault]
    ev = live_evidence(ns)

    os.environ["ESCALATION_BACKEND"] = "ollama"
    os.environ["ESCALATION_MODEL"] = "qwen2.5-coder:14b"

    os.environ["ESCALATION_MODE"] = "single_shot"
    t0 = time.time(); single = escalation.escalate(rc, ev, ns); t_single = time.time() - t0

    os.environ["ESCALATION_MODE"] = "agentic"
    t0 = time.time(); agentic = escalation.escalate(rc, ev, ns); t_agentic = time.time() - t0

    cleanup(ns)
    row = {"fault": fault, "ns": ns,
           "single": {**score_plan(single), "latency_s": round(t_single, 1)},
           "agentic": {**score_plan(agentic), "latency_s": round(t_agentic, 1)}}
    s, a = row["single"], row["agentic"]
    print(f"  [{fault} #{idx}] single: exec={s['executable']} no_ph={s['no_placeholder']} "
          f"steps={s['steps']} ({s['latency_s']}s) | agentic: exec={a['executable']} "
          f"no_ph={a['no_placeholder']} steps={a['steps']} ({a['latency_s']}s)")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--faults", default="oom,image,crashloop")
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--base", default="aiops-planner")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    faults = [f.strip() for f in args.faults.split(",") if f.strip() in FAULTS]
    print(f"[E4] planner agéntico vs single-shot · {faults} · repeat={args.repeat}\n")
    rows = []
    for fault in faults:
        for i in range(args.repeat):
            try:
                rows.append(run_one(fault, args.base, i))
            except Exception as e:
                print(f"  [{fault} #{i}] ERROR: {str(e)[:100]}")

    def rate(mode: str, key: str) -> str:
        vals = [r[mode][key] for r in rows]
        return f"{sum(bool(v) for v in vals)}/{len(vals)}" if vals else "0/0"

    print(f"\n[resumen] n={len(rows)}")
    for mode in ("single", "agentic"):
        print(f"  {mode:8s}: ejecutable {rate(mode,'executable')} · "
              f"sin-placeholder {rate(mode,'no_placeholder')} · seguro {rate(mode,'safe')}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"[resultados] {args.out}")


if __name__ == "__main__":
    main()
