#!/usr/bin/env python3
"""Inyección de caos para evaluar el bucle en producción (E3).

Inyecta un fallo CONOCIDO en un namespace de pruebas aislado, mide la **latencia
de detección** (t_inyección → incidente) y la **corrección del diagnóstico**
(keyword match vs el fallo inyectado), y limpia. Pensado para correr EN EL
SERVIDOR (kubectl + app viva + mismo reloj que el incident store).

Seguro por diseño: solo crea/borra cargas en un namespace dedicado (default
`aiops-chaos`); no toca cargas reales. La parte de remediación/MTTR (que ejecuta
acciones) se deja como paso aparte deliberado.

Uso:
    python eval/chaos_runner.py --fault crashloop
    python eval/chaos_runner.py --fault all --repeat 3
    python eval/chaos_runner.py --fault oom --namespace aiops-chaos --timeout 240
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from eval.metrics import keyword_hit  # noqa: E402

# fault → (scenario_id para keyword_hit, plantilla de manifiesto)
FAULTS = {
    "crashloop": ("crash_config", """
apiVersion: apps/v1
kind: Deployment
metadata: {{name: {name}, namespace: {ns}}}
spec:
  replicas: 1
  selector: {{matchLabels: {{app: {name}}}}}
  template:
    metadata: {{labels: {{app: {name}}}}}
    spec:
      containers:
      - name: c
        image: busybox:1.36
        command: ["sh","-c","echo 'starting bad config'; sleep 2; exit 1"]
"""),
    "oom": ("crash_oom", """
apiVersion: apps/v1
kind: Deployment
metadata: {{name: {name}, namespace: {ns}}}
spec:
  replicas: 1
  selector: {{matchLabels: {{app: {name}}}}}
  template:
    metadata: {{labels: {{app: {name}}}}}
    spec:
      containers:
      - name: c
        image: busybox:1.36
        command: ["sh","-c","tail /dev/zero"]
        resources: {{limits: {{memory: "16Mi"}}, requests: {{memory: "16Mi"}}}}
"""),
    "image": ("image_not_found", """
apiVersion: apps/v1
kind: Deployment
metadata: {{name: {name}, namespace: {ns}}}
spec:
  replicas: 1
  selector: {{matchLabels: {{app: {name}}}}}
  template:
    metadata: {{labels: {{app: {name}}}}}
    spec:
      containers:
      - name: c
        image: registry.invalid/does-not-exist:nope
"""),
}


def kubectl(args: list[str], stdin: str | None = None) -> tuple[str, int]:
    p = subprocess.run(["kubectl", *args], input=stdin, text=True,
                       capture_output=True, timeout=60)
    return ((p.stdout or "") + (p.stderr or ""), p.returncode)


def ensure_ns(ns: str) -> None:
    kubectl(["create", "namespace", ns])  # idempotente; ignora si existe


def inject(fault: str, ns: str, name: str) -> None:
    _, manifest = FAULTS[fault]
    out, rc = kubectl(["apply", "-f", "-"], stdin=manifest.format(name=name, ns=ns))
    if rc != 0:
        raise SystemExit(f"fallo al inyectar: {out}")


def cleanup(ns: str) -> None:
    # Borra el namespace entero (y con él la carga) — async, sin bloquear.
    kubectl(["delete", "namespace", ns, "--ignore-not-found", "--wait=false"])


def wait_for_incident(api: str, ns: str, t0: float, timeout: float, poll: float) -> dict | None:
    deadline = t0 + timeout
    while time.time() < deadline:
        try:
            data = httpx.get(f"{api}/api/incidents", timeout=10).json()
        except Exception:
            time.sleep(poll); continue
        cands = [i for i in data.get("incidents", [])
                 if ns in (i.get("namespaces") or []) and i.get("created_at", 0) >= t0 - 2]
        if cands:
            return sorted(cands, key=lambda i: i.get("created_at", 0))[0]
        time.sleep(poll)
    return None


def run_one(fault: str, base: str, api: str, timeout: float, poll: float, idx: int) -> dict:
    scenario_id, _ = FAULTS[fault]
    # Namespace ÚNICO por inyección: evita la deduplicación por namespace, que si
    # no colapsaría inyecciones repetidas en una sola incidencia.
    ns = f"{base}-{fault}-{idx}"
    name = f"chaos-{fault}"
    ensure_ns(ns)
    t0 = time.time()
    inject(fault, ns, name)
    print(f"  [{fault} #{idx}] inyectado en {ns}/{name}; esperando detección…")
    inc = wait_for_incident(api, ns, t0, timeout, poll)
    if inc is None:
        cleanup(ns)
        print(f"    sin detección en {timeout:.0f}s")
        return {"fault": fault, "scenario_id": scenario_id, "namespace": ns, "detected": False,
                "detection_latency_s": None, "keyword_hit": None, "root_cause": None}
    latency = inc.get("created_at", t0) - t0
    rc = inc.get("root_cause", "")
    kw = keyword_hit(rc, scenario_id)
    print(f"    detectado en {latency:.0f}s · keyword={'✓' if kw else '✗'} · «{rc[:90]}»")
    cleanup(ns)
    return {"fault": fault, "scenario_id": scenario_id, "namespace": ns, "detected": True,
            "detection_latency_s": round(latency, 1), "keyword_hit": bool(kw),
            "incident_id": inc.get("id"), "category": inc.get("category"),
            "root_cause": rc}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", default="crashloop", choices=[*FAULTS, "all"])
    ap.add_argument("--namespace", default="aiops-chaos")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=240.0)
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    faults = list(FAULTS) if args.fault == "all" else [args.fault]
    print(f"[E3] caos · prefijo ns={args.namespace} · fallos={faults} · repeat={args.repeat}\n")
    rows = []
    for fault in faults:
        for i in range(args.repeat):
            rows.append(run_one(fault, args.namespace, args.api, args.timeout, args.poll, i))

    det = [r for r in rows if r["detected"]]
    print(f"\n[resumen] detectados {len(det)}/{len(rows)}")
    if det:
        import statistics
        lat = [r["detection_latency_s"] for r in det]
        kw = sum(r["keyword_hit"] for r in det)
        print(f"  latencia detección: media {statistics.mean(lat):.0f}s · max {max(lat):.0f}s")
        print(f"  diagnóstico correcto (keyword): {kw}/{len(det)}")
    if args.out:
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(f"[resultados] {args.out}")


if __name__ == "__main__":
    main()
