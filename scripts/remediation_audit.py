#!/usr/bin/env python3
"""Histórico de remediaciones + verificación contra el cluster.

  python scripts/remediation_audit.py                 # lista el histórico
  python scripts/remediation_audit.py --verify         # verifica el último rollout
  python scripts/remediation_audit.py --verify INC-XXX # verifica el de un incidente

El histórico se lee de `data/remediation/audit.jsonl` (append-only, durable). La
verificación consulta el cluster en read-only: para un `rollout restart` comprueba
la anotación `kubectl.kubernetes.io/restartedAt` del workload y su `rollout status`,
para confirmar que el reinicio se ejecutó realmente.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.remediation.audit_log import get_audit  # noqa: E402


def _ago(ts: float) -> str:
    s = max(0, int(dt.datetime.now().timestamp() - ts))
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}min"
    return f"{s // 3600}h"


def _kubectl(args: list[str]) -> tuple[str, int]:
    p = subprocess.run(["kubectl", *args], capture_output=True, text=True, timeout=15)
    return ((p.stdout or "") + (p.stderr or "")).strip(), p.returncode


def list_history(records: list[dict], limit: int) -> None:
    if not records:
        print("Sin remediaciones registradas todavía (data/remediation/audit.jsonl).")
        return
    icon = {"done": "✓", "failed": "✗", "manual": "•"}
    print(f"{'cuándo':>7}  {'estado':<7} {'incidente':<14} {'ns':<16} comando")
    print("-" * 100)
    for r in records[-limit:]:
        when = dt.datetime.fromtimestamp(r.get("ts", 0)).strftime("%m-%d %H:%M")
        st = f"{icon.get(r.get('status'), '?')} {r.get('status', '')}"
        print(f"{when}  {st:<7} {r.get('incident_id', ''):<14} "
              f"{r.get('namespace', ''):<16} {r.get('command', '')[:60]}")
    print(f"\n{len(records)} acciones registradas · "
          f"{sum(r.get('status') == 'done' for r in records)} done · "
          f"{sum(r.get('status') == 'failed' for r in records)} failed · "
          f"{sum(r.get('status') == 'manual' for r in records)} manual")


def verify(records: list[dict], incident_id: str | None) -> None:
    rollouts = [r for r in records if "rollout restart" in (r.get("command") or "")
                and (incident_id is None or r.get("incident_id") == incident_id)]
    if not rollouts:
        print("No hay rollout restart registrado" + (f" para {incident_id}" if incident_id else ""))
        return
    r = rollouts[-1]
    cmd, ns = r.get("command", ""), r.get("namespace", "")
    m = re.search(r"rollout\s+restart\s+(\S+)", cmd)
    target = m.group(1) if m else None
    print(f"Incidente {r.get('incident_id')} · {dt.datetime.fromtimestamp(r['ts'])}")
    print(f"  registrado: {r.get('status')} · {cmd}")
    if r.get("status") != "done":
        print(f"  → el audit dice '{r.get('status')}' (no se ejecutó un reinicio); nada que verificar en cluster.")
        return
    if not target:
        print("  → no se pudo extraer el workload del comando."); return
    ann, rc = _kubectl(["get", target, "-n", ns, "-o",
                        "jsonpath={.spec.template.metadata.annotations.kubectl\\.kubernetes\\.io/restartedAt}"])
    if rc != 0:
        print(f"  → no se pudo consultar {target} en {ns}: {ann[:120]}"); return
    print(f"  cluster · restartedAt = {ann or '(sin anotación)'}")
    status, _ = _kubectl(["rollout", "status", target, "-n", ns, "--timeout=8s"])
    print(f"  cluster · rollout status: {status[:120]}")
    if ann:
        print("  ✅ CONFIRMADO: el workload registra un reinicio (restartedAt presente).")
    else:
        print("  ⚠ sin anotación restartedAt — puede no haberse ejecutado por esta vía o el workload se recreó.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", nargs="?", const="__latest__", default=None,
                    metavar="INCIDENT_ID", help="verifica el rollout (último, o de un incidente)")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    records = get_audit().read_all()
    if args.verify is None:
        list_history(records, args.limit)
    else:
        verify(records, None if args.verify == "__latest__" else args.verify)


if __name__ == "__main__":
    main()
