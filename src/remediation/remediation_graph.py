"""
Grafo de conocimiento de remediación.

Memoria NO-paramétrica, estructurada y ejecutable: en vez de mapear
`intent → UNA acción` (como el catálogo de command_builder), guarda por cada
firma de problema un **plan multi-paso** (investigar → identificar → arreglar →
verificar). Resuelve el caso real en que un solo comando no soluciona (p. ej.
ingress: reiniciar haproxy no arregla un backend caído ni una NetworkPolicy).

Diseño (capa abstracta + binding, estilo AST):
  - Nodo = firma abstracta de problema (intent + clase de workload), PORTABLE.
  - Arista = paso de remediación con plantilla de acción (placeholders {ns},
    {pod}, {workload}, {service}, {pvc}, {node}), riesgo y origen.
  - El binding (namespace/recurso reales) se resuelve en runtime con los
    extractores deterministas de command_builder.

La recuperación la hace CÓDIGO (detect_intent + lookup), no el SLM — así el
modelo pequeño nunca carga el grafo en su contexto.

Fase 1: store SQLite + semilla desde el catálogo + resolve con binding. El
escalado a modelo grande (miss), la verificación por outcome y la consolidación
ORPO se añaden en fases posteriores (add_provisional/mark_verified ya stubbeados).
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.diagnostics.command_builder import (
    extract_node,
    extract_pod,
    extract_pvc,
    extract_service,
    extract_workload,
    intent_for,
)

_DEFAULT_DB = os.getenv(
    "AIOPS_GRAPH_DB", "data/graph/remediation_graph.db"
)
# Planes enseñados por un humano: config-as-code versionable en git (el .db SQLite
# es binario y no versiona bien). El grafo se siembra de este YAML al arrancar.
_TAUGHT_YAML = os.getenv("AIOPS_TAUGHT_PLANS", "data/graph/taught_plans.yml")

# Tipos de paso
INVESTIGATE = "investigate"
COMMAND = "command"      # acción de escritura reversible (shadow + aprobación)
GUIDANCE = "guidance"    # acción manual (texto), sin comando seguro
VERIFY = "verify"

# Origen de un paso/nodo
SOURCE_CATALOG = "catalog"      # semilla determinista
SOURCE_ESCALATED = "escalated"  # propuesto por el modelo grande (provisional)
SOURCE_HUMAN = "human"          # enseñado por un operador (verificado al instante)


@dataclass
class Step:
    order: int
    action_type: str
    action: str            # comando kubectl ya enlazado, o texto de guía
    explanation: str = ""
    risk_level: int = 0    # 0 lectura · 1 reversible · 3 destructivo
    source: str = "catalog"
    verified: bool = False


@dataclass
class Plan:
    intent: str
    namespace: str
    steps: list[Step] = field(default_factory=list)
    source: str = "graph"

    def to_dicts(self) -> list[dict]:
        return [asdict(s) for s in self.steps]


# ── Semilla: plan multi-paso por intención (capa abstracta) ──────────────────
# Cada paso: (action_type, action_template, explanation, risk_level)
# Las plantillas con {pod}/{workload}/{service}/{pvc}/{node} se descartan si el
# recurso no se puede extraer de la evidencia (el plan conserva los pasos de {ns}).

_SEED_PLANS: dict[str, dict] = {
    "network": {
        "namespace_class": "ingress-controller",
        "steps": [
            (INVESTIGATE, "kubectl get endpoints -n {ns}",
             "¿El service tiene endpoints (backend Ready)? Si no, el fallo está en el "
             "backend, no en el ingress.", 0),
            (INVESTIGATE, "kubectl describe ingress -n {ns}",
             "Revisa host/path/backend de la regla de Ingress por si está mal configurada.", 0),
            (INVESTIGATE, "kubectl get networkpolicy -n {ns}",
             "Comprueba si una NetworkPolicy está bloqueando el tráfico del namespace.", 0),
            (COMMAND, "kubectl rollout restart deployment/{workload} -n {ns}",
             "Último recurso reversible: reinicia el controlador si su estado está colgado.", 1),
        ],
    },
    "endpoints": {
        "namespace_class": "service",
        "steps": [
            (INVESTIGATE, "kubectl get endpoints -n {ns}",
             "¿Hay pods Ready detrás del service?", 0),
            (INVESTIGATE, "kubectl describe pods -n {ns}",
             "Revisa por qué los pods no están Ready (readiness/selector).", 0),
            (COMMAND, "kubectl rollout restart deployment/{workload} -n {ns}",
             "Reinicia el backend para que vuelva a registrarse en el service.", 1),
        ],
    },
    "crash_secret": {
        "namespace_class": "app",
        "steps": [
            (INVESTIGATE, "kubectl get secret -n {ns}",
             "¿Existe el secret con el rol/credenciales que la app no encuentra?", 0),
            (COMMAND, "kubectl rollout restart deployment/{workload} -n {ns}",
             "Reinicia el workload para que recoja el secret corregido.", 1),
        ],
    },
    "crash_config": {
        "namespace_class": "app",
        "steps": [
            (INVESTIGATE, "kubectl logs {pod} -n {ns} --previous",
             "Mira el log de la instancia que crasheó para ver el error de arranque.", 0),
            (COMMAND, "kubectl rollout restart deployment/{workload} -n {ns}",
             "Reinicia el workload tras corregir la configuración.", 1),
        ],
    },
    "oom": {
        "namespace_class": "app",
        "steps": [
            (INVESTIGATE, "kubectl describe pod {pod} -n {ns}",
             "Confirma OOMKilled y el límite de memoria del contenedor.", 0),
            (COMMAND, "kubectl rollout restart deployment/{workload} -n {ns}",
             "Reinicia para aplicar el nuevo límite de memoria.", 1),
        ],
    },
    "probe": {
        "namespace_class": "app",
        "steps": [
            (INVESTIGATE, "kubectl describe pod {pod} -n {ns}",
             "Revisa el evento de liveness/readiness probe.", 0),
            (COMMAND, "kubectl rollout restart deployment/{workload} -n {ns}",
             "Reinicia tras ajustar el probe.", 1),
        ],
    },
    "image": {
        "namespace_class": "app",
        "steps": [
            (INVESTIGATE, "kubectl describe pod {pod} -n {ns}",
             "Mira el evento de pull para ver si es tag, registry o autenticación.", 0),
        ],
    },
    "image_auth": {
        "namespace_class": "app",
        "steps": [
            (INVESTIGATE, "kubectl get secret -n {ns}",
             "¿Existe el secret de pull referenciado en imagePullSecrets?", 0),
        ],
    },
    "pvc": {
        "namespace_class": "storage",
        "steps": [
            (INVESTIGATE, "kubectl describe pvc {pvc} -n {ns}",
             "Mira por qué el PVC no se vincula a un volumen.", 0),
        ],
    },
    "node_pressure": {
        "namespace_class": "node",
        "steps": [
            (INVESTIGATE, "kubectl describe node {node}",
             "Revisa la presión de memoria/disco y las condiciones del nodo.", 0),
        ],
    },
    "pending_cpu": {
        "namespace_class": "app",
        "steps": [
            (INVESTIGATE, "kubectl describe pod {pod} -n {ns}",
             "Mira por qué no se planifica (FailedScheduling / Insufficient cpu).", 0),
        ],
    },
}

_PLACEHOLDER_RE = re.compile(r"\{[a-z_]+\}")


def _bind(template: str, ns: str, evidence: str) -> str | None:
    """Sustituye placeholders con recursos extraídos de la evidencia.

    Devuelve None si queda algún placeholder sin resolver (el paso se descarta),
    salvo {ns} que siempre debe poder resolverse para pasos namespaced.
    """
    out = template
    repl = {
        "{ns}": ns or "",
        "{pod}": extract_pod(evidence) or "",
        "{workload}": extract_workload(evidence) or "",
        "{service}": extract_service(evidence) or "",
        "{pvc}": extract_pvc(evidence) or "",
        "{node}": extract_node(evidence) or "",
    }
    for k, v in repl.items():
        if k in out:
            if not v:
                return None  # recurso necesario no disponible → descartar paso
            out = out.replace(k, v)
    if _PLACEHOLDER_RE.search(out):
        return None
    return re.sub(r"\s+", " ", out).strip()


def _embed_text(text: str) -> list[float] | None:
    """Embedding (nomic-embed) del texto de firma; None si Ollama no disponible."""
    try:
        from src.diagnostics.semantic_ground import _embed
        return _embed((text or "")[:1000])
    except Exception:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _taught_path(path: str | None = None) -> str:
    """Ruta del YAML de planes enseñados, resuelta en cada llamada (env-driven)."""
    return path or os.getenv("AIOPS_TAUGHT_PLANS", _TAUGHT_YAML)


def _load_taught_yaml(path: str | None = None) -> list[dict]:
    """Lee los planes enseñados del YAML versionado. [] si no existe/está vacío."""
    p = Path(_taught_path(path))
    if not p.exists():
        return []
    try:
        import yaml
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    plans = data.get("plans") if isinstance(data, dict) else None
    return [pl for pl in (plans or []) if isinstance(pl, dict)]


def _append_taught_yaml(plan: dict, path: str | None = None) -> None:
    """Añade un plan al YAML (config-as-code) preservando los ya existentes.

    Es idempotente por `intent`: si ya existe una entrada con ese intent, la
    reemplaza (así una re-enseñanza corrige en vez de duplicar)."""
    import yaml
    path = _taught_path(path)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_taught_yaml(path)
    existing = [e for e in existing if e.get("intent") != plan.get("intent")]
    existing.append(plan)
    header = (
        "# Planes de remediación enseñados por operadores (config-as-code).\n"
        "# Versionar en git: es la fuente de verdad; el grafo SQLite se siembra\n"
        "# de aquí al arrancar (seed_from_taught) y al enseñar (activo al instante).\n"
    )
    body = yaml.safe_dump({"plans": existing}, allow_unicode=True, sort_keys=False)
    p.write_text(header + body, encoding="utf-8")


class RemediationGraph:
    """Store SQLite del grafo de remediación (nodes + edges)."""

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent TEXT NOT NULL UNIQUE,
                namespace_class TEXT DEFAULT '',
                label TEXT DEFAULT '',
                signature_text TEXT DEFAULT '',
                embedding TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                step_order INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                action_template TEXT NOT NULL,
                explanation TEXT DEFAULT '',
                risk_level INTEGER DEFAULT 0,
                source TEXT DEFAULT 'catalog',
                verified INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                attempt_count INTEGER DEFAULT 0,
                UNIQUE(node_id, step_order, source)
            );
            """
        )
        self._conn.commit()

    def seed_from_catalog(self) -> None:
        """Siembra el grafo desde el catálogo (idempotente). El grafo arranca
        lleno con todo lo que el sistema ya sabía hacer → cero regresión."""
        cur = self._conn.cursor()
        for intent, spec in _SEED_PLANS.items():
            cur.execute(
                "INSERT OR IGNORE INTO nodes(intent, namespace_class, label) VALUES (?,?,?)",
                (intent, spec.get("namespace_class", ""), intent),
            )
            row = cur.execute("SELECT id FROM nodes WHERE intent=?", (intent,)).fetchone()
            node_id = row["id"]
            for i, (atype, tmpl, expl, risk) in enumerate(spec["steps"]):
                cur.execute(
                    "INSERT OR IGNORE INTO edges(node_id, step_order, action_type, "
                    "action_template, explanation, risk_level, source) VALUES (?,?,?,?,?,?,?)",
                    (node_id, i, atype, tmpl, expl, risk, "catalog"),
                )
        self._conn.commit()

    def resolve(self, evidence: str, namespace: str, root_cause: str = "") -> Plan | None:
        """Devuelve un plan multi-paso para la firma del problema, o None (miss).

        La intención se detecta con el detector determinista de command_builder
        (mismo que el resto del sistema). El binding usa los extractores.
        """
        # 1) clave determinista por intención (hits del catálogo).
        node_id = None
        intent_name = ""
        intent = intent_for(root_cause, evidence)
        if intent is not None:
            row = self._conn.execute(
                "SELECT id FROM nodes WHERE intent=?", (intent["name"],)
            ).fetchone()
            if row is not None:
                node_id, intent_name = row["id"], intent["name"]
        # 2) si no hay intención conocida, vecino más cercano por embedding
        #    (nodos escalados de problemas novedosos).
        if node_id is None:
            node_id, intent_name = self._embedding_nearest(f"{root_cause}\n{evidence}")
        if node_id is None:
            return None
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE node_id=? ORDER BY step_order", (node_id,)
        ).fetchall()
        ns = (namespace or "").strip()
        steps: list[Step] = []
        order = 0
        for r in rows:
            if r["action_type"] == GUIDANCE:
                continue  # los pasos manuales se han retirado (incl. nodos antiguos en BD)
            action = _bind(r["action_template"], ns, evidence)
            if action is None:
                continue  # recurso no disponible → descartar el paso
            steps.append(Step(
                order=order,
                action_type=r["action_type"],
                action=action,
                explanation=r["explanation"],
                risk_level=r["risk_level"],
                source=r["source"],
                verified=bool(r["verified"]),
            ))
            order += 1
        if not steps:
            return None
        return Plan(intent=intent_name, namespace=ns, steps=steps, source="graph")

    def _embedding_nearest(self, query: str, threshold: float = 0.6) -> tuple:
        """Vecino más cercano por embedding sobre nodos con embedding almacenado
        (escalados). Devuelve (node_id, intent) o (None, '')."""
        q = _embed_text(query)
        if q is None:
            return (None, "")
        best_id, best_name, best = None, "", threshold
        for row in self._conn.execute(
            "SELECT id, intent, embedding FROM nodes WHERE embedding != ''"
        ).fetchall():
            try:
                emb = json.loads(row["embedding"])
            except (ValueError, TypeError):
                continue
            sc = _cosine(q, emb)
            if sc >= best:
                best, best_id, best_name = sc, row["id"], row["intent"]
        return (best_id, best_name)

    def stats(self) -> dict:
        n = self._conn.execute("SELECT COUNT(*) c FROM nodes").fetchone()["c"]
        e = self._conn.execute("SELECT COUNT(*) c FROM edges").fetchone()["c"]
        v = self._conn.execute(
            "SELECT COUNT(DISTINCT node_id) c FROM edges WHERE verified=1"
        ).fetchone()["c"]
        return {"nodes": n, "edges": e, "verified": v}

    def dump(self) -> list[dict]:
        """Vuelca el grafo (read-only) para visualización: cada nodo con sus pasos.

        `source` del nodo se deriva de sus aristas: 'catalog' si todas vienen del
        catálogo, 'escalated' si alguna fue propuesta por el modelo grande.
        """
        out: list[dict] = []
        nodes = self._conn.execute(
            "SELECT id, intent, namespace_class, label FROM nodes ORDER BY intent"
        ).fetchall()
        for node in nodes:
            rows = self._conn.execute(
                "SELECT step_order, action_type, action_template, explanation, "
                "risk_level, source, verified, success_count, attempt_count "
                "FROM edges WHERE node_id=? ORDER BY step_order",
                (node["id"],),
            ).fetchall()
            steps = [
                {
                    "order": r["step_order"],
                    "action_type": r["action_type"],
                    "action": r["action_template"],
                    "explanation": r["explanation"],
                    "risk_level": r["risk_level"],
                    "source": r["source"],
                    "verified": bool(r["verified"]),
                    "success_count": r["success_count"],
                    "attempt_count": r["attempt_count"],
                }
                for r in rows
                if r["action_type"] != GUIDANCE  # los pasos manuales se han retirado
            ]
            sources = {s["source"] for s in steps}
            node_source = (
                SOURCE_HUMAN if SOURCE_HUMAN in sources
                else SOURCE_ESCALATED if SOURCE_ESCALATED in sources
                else SOURCE_CATALOG
            )
            out.append({
                "intent": node["intent"],
                "namespace_class": node["namespace_class"],
                "label": node["label"],
                "source": node_source,
                "verified": any(s["verified"] for s in steps),
                "steps": steps,
            })
        return out

    def add_provisional(self, key: str, steps: list[Step], signature_text: str = "",
                        namespace_class: str = "") -> None:
        """Fase 2: escribe un plan propuesto por el modelo grande (sin verificar),
        con un embedding de su firma para poder reencontrarlo en futuros miss."""
        if not steps:
            return
        emb = _embed_text(signature_text) if signature_text else None
        emb_json = json.dumps(emb) if emb else ""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO nodes(intent, namespace_class, label, signature_text, "
            "embedding) VALUES (?,?,?,?,?)",
            (key, namespace_class, key, signature_text[:500], emb_json),
        )
        node_id = cur.execute("SELECT id FROM nodes WHERE intent=?", (key,)).fetchone()["id"]
        base = cur.execute(
            "SELECT COALESCE(MAX(step_order),-1)+1 o FROM edges WHERE node_id=?", (node_id,)
        ).fetchone()["o"]
        for j, s in enumerate(steps):
            cur.execute(
                "INSERT OR IGNORE INTO edges(node_id, step_order, action_type, "
                "action_template, explanation, risk_level, source) VALUES (?,?,?,?,?,?,?)",
                (node_id, base + j, s.action_type, s.action, s.explanation,
                 s.risk_level, "escalated"),
            )
        self._conn.commit()

    def add_taught(self, intent: str, steps: list[Step], signature_text: str = "",
                   namespace_class: str = "", label: str = "",
                   persist_yaml: bool = True) -> None:
        """Enseñanza humana: escribe un plan para una firma nueva como nodo
        **verificado** (source='human'), activo al instante.

        A diferencia de `add_provisional` (modelo grande, sin verificar), un plan
        enseñado por un operador se marca verificado directamente — la señal de
        mayor calidad. La seguridad no se relaja: los pasos `command` siguen
        llevando su `risk_level` y pasan por dry-run + aprobación shadow al
        ejecutarse; enseñar solo añade el plan a la memoria.

        Si `persist_yaml`, además vuelca el plan a `taught_plans.yml`
        (config-as-code versionable). El seed de arranque llama con False.
        """
        if not steps:
            return
        emb = _embed_text(signature_text) if signature_text else None
        emb_json = json.dumps(emb) if emb else ""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO nodes(intent, namespace_class, label, signature_text, "
            "embedding) VALUES (?,?,?,?,?)",
            (intent, namespace_class, label or intent, signature_text[:500], emb_json),
        )
        node_id = cur.execute("SELECT id FROM nodes WHERE intent=?", (intent,)).fetchone()["id"]
        # Re-enseñar reemplaza los pasos humanos previos de este nodo (corrige, no duplica).
        cur.execute("DELETE FROM edges WHERE node_id=? AND source=?", (node_id, SOURCE_HUMAN))
        for j, s in enumerate(steps):
            cur.execute(
                "INSERT OR IGNORE INTO edges(node_id, step_order, action_type, "
                "action_template, explanation, risk_level, source, verified) "
                "VALUES (?,?,?,?,?,?,?,1)",
                (node_id, j, s.action_type, s.action, s.explanation, s.risk_level,
                 SOURCE_HUMAN),
            )
        self._conn.commit()
        if persist_yaml:
            _append_taught_yaml({
                "intent": intent,
                "namespace_class": namespace_class,
                "label": label or intent,
                "signature_text": signature_text[:500],
                "steps": [
                    {"type": s.action_type, "action": s.action,
                     "explanation": s.explanation, "risk": s.risk_level}
                    for s in steps
                ],
            })

    def seed_from_taught(self) -> None:
        """Siembra el grafo desde `taught_plans.yml` (idempotente). El YAML es la
        fuente de verdad versionada; esto lo indexa en SQLite al arrancar."""
        for pl in _load_taught_yaml():
            intent = str(pl.get("intent", "")).strip()
            raw_steps = pl.get("steps") or []
            if not intent or not isinstance(raw_steps, list):
                continue
            steps = [
                Step(
                    order=i,
                    action_type=str(st.get("type", "")).strip(),
                    action=str(st.get("action", "")).strip(),
                    explanation=str(st.get("explanation", "")).strip(),
                    risk_level=int(st.get("risk", 0) or 0),
                    source=SOURCE_HUMAN,
                    verified=True,
                )
                for i, st in enumerate(raw_steps) if isinstance(st, dict)
            ]
            self.add_taught(
                intent, steps,
                signature_text=str(pl.get("signature_text", "")),
                namespace_class=str(pl.get("namespace_class", "")),
                label=str(pl.get("label", "")),
                persist_yaml=False,
            )

    def mark_verified(self, intent: str, success: bool) -> None:
        """Fase 3: marca las aristas del intent como verificadas por outcome."""
        node = self._conn.execute("SELECT id FROM nodes WHERE intent=?", (intent,)).fetchone()
        if node is None:
            return
        self._conn.execute(
            "UPDATE edges SET attempt_count=attempt_count+1, "
            "success_count=success_count+?, verified=CASE WHEN ? THEN 1 ELSE verified END "
            "WHERE node_id=?",
            (1 if success else 0, 1 if success else 0, node["id"]),
        )
        self._conn.commit()


def verify_from_incident(graph: RemediationGraph, inc: dict) -> bool | None:
    """Fase 3 — verificación por outcome (señal existente).

    Cuando un incidente cuyo plan vino del grafo llega a un estado terminal,
    marca su nodo como verificado (éxito) o registra el intento (fallo).
    Devuelve True/False según la señal, o None si no aplica.
    """
    if inc.get("solution_source") not in ("graph", "escalated"):
        return None
    key = inc.get("solution_key")
    if not key:
        return None
    status = inc.get("status")
    resp = inc.get("response")
    verified = inc.get("verified")
    if status == "resolved" and (verified or resp == "approved"):
        graph.mark_verified(key, True)
        return True
    if status == "failed" or resp == "rejected":
        graph.mark_verified(key, False)
        return False
    return None


_GRAPH: RemediationGraph | None = None


def get_graph() -> RemediationGraph:
    """Singleton perezoso: crea el store y lo siembra del catálogo una vez."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = RemediationGraph()
        _GRAPH.seed_from_catalog()
        _GRAPH.seed_from_taught()
    return _GRAPH
