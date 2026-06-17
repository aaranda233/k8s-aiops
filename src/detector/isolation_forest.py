"""
Capa 2 — Deteccion de anomalias con Isolation Forest.

Estrategia de entrenamiento (Enfoque C):
  1. BOOTSTRAP: acumular N ventanas iniciales asumiendo comportamiento normal.
     El modelo se entrena por primera vez con ese baseline.
  2. DETECCION: cada ventana nueva se puntua contra el modelo actual.
  3. REENTRENAMIENTO PERIODICO: cada K ventanas nuevas, el modelo se reentrena
     con las ultimas M ventanas (ventana deslizante), descartando historia antigua.
     Esto permite adaptarse al drift sin olvidar patrones aprendidos.

Por que funciona sin etiquetas:
  - Las anomalias son estadisticamente escasas → el IF las aísla con pocos cortes.
  - El baseline normal es denso → necesita muchos cortes para aislarse.
  - No asumimos CUALES son anomalias, solo que son poco frecuentes (contamination %).
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize

from src.detector.window import WindowData

# Señal de severidad: un namespace donde una proporción anormal de SUS logs son
# de error (FATAL/ERROR/CRITICAL) es anómalo, aunque su volumen sea pequeño frente
# al cluster. Se evalúa POR NAMESPACE (no global) para que un servicio "callado"
# pero ardiendo no se diluya entre el ruido de todo el cluster — antes solo
# disparaban los namespaces ruidosos (postgresql) y se escapaban los demás.
_SEVERITY_MIN_ERRORS = 3    # mínimo de logs de error EN UN namespace
_SEVERITY_LOW = 0.25        # ratio de error LOCAL donde empieza a puntuar
_SEVERITY_HIGH = 0.70       # ratio de error LOCAL donde satura a 1.0


def _map_severity(total: int, err: int) -> float:
    """Mapea (total, errores) de UN namespace a un sub-score [0,1]."""
    if err < _SEVERITY_MIN_ERRORS or total == 0:
        return 0.0
    ratio = err / total
    if ratio <= _SEVERITY_LOW:
        return 0.0
    return min(1.0, (ratio - _SEVERITY_LOW) / (_SEVERITY_HIGH - _SEVERITY_LOW))


def severity_by_namespace(window: WindowData) -> dict[str, float]:
    """Severidad [0,1] por namespace (proporción local de logs de error)."""
    ns_log = getattr(window, "ns_log_counts", None)
    ns_err = getattr(window, "ns_error_counts", None)
    if not ns_log:
        return {}
    out: dict[str, float] = {}
    for ns, total in ns_log.items():
        s = _map_severity(total, (ns_err or {}).get(ns, 0))
        if s > 0.0:
            out[ns] = s
    return out


def severity_score(window: WindowData) -> float:
    """Sub-score [0,1] = máximo sobre namespaces de su proporción local de errores."""
    by_ns = severity_by_namespace(window)
    if by_ns:
        return max(by_ns.values())
    # Compatibilidad con ventanas sin desglose por namespace: ratio global.
    if window.error_count < _SEVERITY_MIN_ERRORS:
        return 0.0
    ratio = window.error_ratio
    return 0.0 if ratio <= _SEVERITY_LOW else min(1.0, (ratio - _SEVERITY_LOW) / (_SEVERITY_HIGH - _SEVERITY_LOW))


# Señal de novedad: una ventana con muchos logs de plantillas NUNCA vistas por el
# modelo (no están en el feature set entrenado) es anómala — un patrón de log
# inédito es la señal más fuerte de "algo nuevo está pasando". El IF ignora las
# plantillas nuevas hasta el siguiente reentrenamiento (su feature set está
# congelado), así que por sí solo no las detecta; esta señal cierra ese hueco.
# Es transitoria: cuando el modelo reentrena e incorpora la plantilla, deja de
# ser nueva y toman el relevo la distribución (IF) y la severidad.
_NOVELTY_MIN_LOGS = 5    # mínimo de logs novedosos para considerarlo
_NOVELTY_LOW = 0.15      # ratio de novedad donde empieza a puntuar
_NOVELTY_HIGH = 0.50     # ratio de novedad donde satura a 1.0


def _map_novelty(novel: int, total: int) -> float:
    """Mapea (logs novedosos, total) de UN namespace a un sub-score [0,1]."""
    if total == 0 or novel < _NOVELTY_MIN_LOGS:
        return 0.0
    ratio = novel / total
    if ratio <= _NOVELTY_LOW:
        return 0.0
    return min(1.0, (ratio - _NOVELTY_LOW) / (_NOVELTY_HIGH - _NOVELTY_LOW))


def _novelty_from_counts(counts: dict[int, int], trained_ids: set[int]) -> float:
    total = sum(counts.values())
    novel = sum(c for cid, c in counts.items() if cid not in trained_ids)
    return _map_novelty(novel, total)


def novelty_by_namespace(window: WindowData, trained_ids: set[int]) -> dict[str, float]:
    """Novedad [0,1] por namespace (proporción de logs de plantillas no vistas)."""
    ns_counts = getattr(window, "ns_cluster_counts", None)
    if not ns_counts:
        return {}
    out: dict[str, float] = {}
    for ns, counts in ns_counts.items():
        s = _novelty_from_counts(counts, trained_ids)
        if s > 0.0:
            out[ns] = s
    return out


def novelty_score(window: WindowData, trained_ids: set[int]) -> float:
    """Sub-score [0,1] por proporción de logs de plantillas no vistas (whole-window)."""
    return _novelty_from_counts(window.cluster_counts, trained_ids)


@dataclass
class ScoredWindow:
    window: WindowData
    score: float          # 0 = normal, 1 = anomalia maxima
    is_anomaly: bool
    model_version: int    # que version del modelo la puntuo
    pca_x: float = 0.0   # coordenada 2D para scatter plot
    pca_y: float = 0.0
    in_training: bool = False  # esta ventana forma parte del training set actual
    severity_score: float = 0.0  # componente por severidad de logs (error_ratio)
    novelty_score: float = 0.0   # componente por plantillas nunca vistas
    if_score: float = 0.0        # componente Isolation Forest (distribución)
    culprit_namespace: str = ""  # namespace que alcanzó el score máximo (foco del RCA)


class AnomalyDetector:
    def __init__(
        self,
        bootstrap_windows: int = 10,
        rolling_window_size: int = 50,
        retrain_every_n: int = 5,
        threshold: float = 0.80,
        n_estimators: int = 200,
        contamination: float = 0.05,
        random_state: int = 42,
        warmup_windows: int = 0,
    ):
        self.bootstrap_windows = bootstrap_windows
        self.rolling_size = rolling_window_size
        self.retrain_every_n = retrain_every_n
        self.threshold = threshold
        # Warm-up: nº de ventanas tras arrancar durante las que la NOVEDAD se
        # amortigua (rampa 0→1). 0 = desactivado. Tras un arranque en frío el
        # parser y el modelo están vacíos y todo template es "nuevo", así que la
        # novedad es ruido; severidad e IF no se tocan.
        self.warmup_windows = warmup_windows
        self._windows_since_ready: int = 0
        self._if_params = dict(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

        self._model: IsolationForest | None = None
        self._model_version: int = 0
        self._trained_cluster_ids: list[int] = []
        # Escala del lado anómalo de decision_function (referencia absoluta)
        self._d_scale: float = 0.05
        # PCA 2D para visualizacion — se recalcula en cada entrenamiento
        self._pca: PCA | None = None
        # Coordenadas 2D de las ventanas de entrenamiento para el scatter
        self._training_coords: list[tuple[float, float]] = []

        # Buffer circular para reentrenamiento (ventana deslizante)
        self._history: deque[WindowData] = deque(maxlen=rolling_window_size)
        # Ventanas acumuladas desde el ultimo reentrenamiento
        self._since_last_retrain: int = 0
        # Conjunto completo de cluster_ids vistos (crece con el tiempo)
        self._all_cluster_ids: set[int] = set()

        # Estado
        self._bootstrapping: bool = True
        self._bootstrap_buffer: list[WindowData] = []

    # ------------------------------------------------------------------
    # API principal
    # ------------------------------------------------------------------

    def process(self, window: WindowData) -> ScoredWindow | None:
        """
        Procesa una ventana nueva.

        Durante bootstrap: acumula ventanas y retorna None.
        Una vez listo: puntua y opcionalmente reentrena.

        Returns:
            ScoredWindow con score y flag de anomalia,
            o None si aun estamos en fase bootstrap.
        """
        self._update_cluster_ids(window)
        self._history.append(window)

        if self._bootstrapping:
            self._bootstrap_buffer.append(window)
            if len(self._bootstrap_buffer) >= self.bootstrap_windows:
                self._train(self._bootstrap_buffer)
                self._bootstrapping = False
            return None, False

        # Puntuar POR NAMESPACE: cada namespace de la ventana se evalúa por
        # separado con tres señales complementarias (IF de distribución, severidad
        # de logs, novedad de plantillas). El score de la ventana es el máximo y el
        # CULPABLE es el namespace que lo alcanza — así un namespace sano no arrastra
        # a toda la ventana a anomalía y la alerta apunta al servicio real.
        pca_coord = self._project_pca(window)
        self._windows_since_ready += 1
        warm = self._novelty_warmup_factor()
        trained = set(self._trained_cluster_ids)

        if getattr(window, "ns_cluster_counts", None):
            if_by_ns = self._if_by_namespace(window)
            nov_by_ns = novelty_by_namespace(window, trained)
        else:
            # Compatibilidad: ventana sin desglose → un único pseudo-namespace.
            if_by_ns = {"": self._score_whole(window)}
            nov_by_ns = {"": novelty_score(window, trained)}
        sev_by_ns = severity_by_namespace(window)

        culprit, score = "", 0.0
        for ns in sorted(set(if_by_ns) | set(sev_by_ns) | set(nov_by_ns)):
            s = max(if_by_ns.get(ns, 0.0), sev_by_ns.get(ns, 0.0),
                    nov_by_ns.get(ns, 0.0) * warm)
            if s > score:
                score, culprit = s, ns

        # Señales a nivel ventana (fuerza máxima cruda, para UI/diagnóstico).
        sev = max(sev_by_ns.values(), default=0.0)
        nov = max(nov_by_ns.values(), default=0.0)
        if_s = max(if_by_ns.values(), default=0.0)
        self._since_last_retrain += 1

        retrained = False
        if self._since_last_retrain >= self.retrain_every_n:
            self._train(list(self._history))
            self._since_last_retrain = 0
            retrained = True

        return ScoredWindow(
            window=window,
            score=score,
            is_anomaly=score >= self.threshold,
            model_version=self._model_version,
            pca_x=pca_coord[0],
            pca_y=pca_coord[1],
            in_training=False,
            severity_score=sev,
            novelty_score=nov,
            if_score=if_s,
            culprit_namespace=culprit,
        ), retrained

    def _novelty_warmup_factor(self) -> float:
        """Factor [0,1] que amortigua la novedad tras (re)arrancar (rampa lineal).

        Nada más arrancar TODO es novedoso (parser/modelo en frío), así que la
        señal de novedad es poco informativa y satura el detector. La rampa la
        reintroduce gradualmente conforme el baseline madura.
        """
        if self.warmup_windows <= 0:
            return 1.0
        return min(1.0, self._windows_since_ready / self.warmup_windows)

    @property
    def is_ready(self) -> bool:
        return not self._bootstrapping

    @property
    def bootstrap_progress(self) -> str:
        n = len(self._bootstrap_buffer)
        return f"{n}/{self.bootstrap_windows} ventanas"

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _update_cluster_ids(self, window: WindowData) -> None:
        self._all_cluster_ids.update(window.cluster_counts.keys())

    def _vectorize(self, windows: list[WindowData], feature_ids: list[int]) -> np.ndarray:
        """
        Convierte lista de WindowData en matriz numerica usando un feature set fijo.

        feature_ids: lista ordenada de cluster_ids que define las columnas.
        Cluster_ids nuevos (no vistos en entrenamiento) se ignoran — columna 0.
        """
        if not feature_ids:
            return np.zeros((len(windows), 1), dtype=np.float32)

        matrix = np.zeros((len(windows), len(feature_ids)), dtype=np.float32)
        for i, w in enumerate(windows):
            for j, cid in enumerate(feature_ids):
                matrix[i, j] = w.cluster_counts.get(cid, 0)

        return normalize(matrix, norm="l1")

    def _vectorize_one(self, counts: dict[int, int], feature_ids: list[int]) -> np.ndarray:
        """Vector l1-normalizado de UNA distribución de plantillas (un namespace)."""
        if not feature_ids:
            return np.zeros(1, dtype=np.float32)
        v = np.zeros(len(feature_ids), dtype=np.float32)
        for j, cid in enumerate(feature_ids):
            v[j] = counts.get(cid, 0)
        s = v.sum()
        return v / s if s > 0 else v

    def _namespace_vectors(self, window: WindowData, feature_ids: list[int]) -> list[np.ndarray]:
        """Vectores por namespace de una ventana (o whole-window si no hay desglose)."""
        nsc = getattr(window, "ns_cluster_counts", None)
        if nsc:
            return [self._vectorize_one(c, feature_ids) for c in nsc.values()]
        return [self._vectorize_one(window.cluster_counts, feature_ids)]

    def _train(self, windows: list[WindowData]) -> None:
        self._trained_cluster_ids = sorted(self._all_cluster_ids)
        ids = self._trained_cluster_ids
        # El IF se entrena sobre vectores POR NAMESPACE (fila = (ventana, namespace)):
        # aprende cómo es la distribución de plantillas de un namespace "normal".
        rows = [v for w in windows for v in self._namespace_vectors(w, ids)]
        X = np.vstack(rows) if rows else np.zeros((1, max(1, len(ids))), dtype=np.float32)
        model = IsolationForest(**self._if_params)
        model.fit(X)
        self._model = model
        self._model_version += 1
        # Escala ABSOLUTA basada en decision_function (calibrada por contamination):
        # d>=0 = dentro de la distribución normal → score 0; d<0 = anómalo. La escala
        # del lado anómalo = profundidad del peor punto normal del entrenamiento.
        # Así una ventana normal NO se fuerza a 1.0 aunque sea la "menos normal" del
        # momento — esa normalización relativa min-max era la causa del flood.
        d_train = model.decision_function(X)
        self._d_scale = float(max(-d_train.min(), 0.05))

        # PCA 2D para visualización: se ajusta sobre el agregado POR VENTANA (no
        # por-namespace) para que el scatter siga teniendo una coordenada por
        # ventana de entrenamiento (consistente con el evento de retrain).
        Xw = self._vectorize(windows, ids)
        n_components = min(2, Xw.shape[0], Xw.shape[1])
        if n_components == 2:
            self._pca = PCA(n_components=2, random_state=42)
            coords = self._pca.fit_transform(Xw)
            self._training_coords = [(float(c[0]), float(c[1])) for c in coords]
        else:
            self._pca = None
            self._training_coords = [(0.0, 0.0)] * len(windows)

    def get_training_scatter(self, windows: list[WindowData]) -> list[tuple[float, float]]:
        """Coordenadas PCA de las ventanas de entrenamiento actuales."""
        return list(self._training_coords)

    def _project_pca(self, window: WindowData) -> tuple[float, float]:
        """Proyecta el vector whole-window al espacio PCA del modelo (solo viz)."""
        if self._pca is None or not self._trained_cluster_ids:
            return (0.0, 0.0)
        v = self._vectorize_one(window.cluster_counts, self._trained_cluster_ids).reshape(1, -1)
        try:
            proj = self._pca.transform(v)[0]
            return (float(proj[0]), float(proj[1]))
        except Exception:
            return (0.0, 0.0)

    def _if_anom(self, d: float) -> float:
        """decision_function → score de anomalía [0,1]. d>=0 (normal) → 0."""
        if d >= 0:
            return 0.0
        return float(np.clip(-d / self._d_scale, 0.0, 1.0))

    def _score_whole(self, window: WindowData) -> float:
        """Score IF [0,1] de la ventana entera (fallback sin desglose por namespace)."""
        if self._model is None or not self._trained_cluster_ids:
            return 0.0
        v = self._vectorize_one(window.cluster_counts, self._trained_cluster_ids)
        d = float(self._model.decision_function(v.reshape(1, -1))[0])
        return self._if_anom(d)

    def _if_by_namespace(self, window: WindowData) -> dict[str, float]:
        """Score IF [0,1] de cada namespace de la ventana (referencia absoluta)."""
        if self._model is None or not self._trained_cluster_ids:
            return {}
        cur = list((getattr(window, "ns_cluster_counts", None) or {}).items())
        if not cur:
            return {}
        ids = self._trained_cluster_ids
        X = np.vstack([self._vectorize_one(c, ids) for _, c in cur])
        d = self._model.decision_function(X)
        return {ns: self._if_anom(float(di)) for (ns, _), di in zip(cur, d)}
