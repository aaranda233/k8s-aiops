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

# Señal de severidad: una ventana con un volumen anormal de logs de error
# (FATAL/ERROR/CRITICAL) es anómala aunque su distribución de plantillas sea
# estadísticamente regular. Captura fallos que solo se ven en los logs, no en
# el estado del pod ni en la mezcla de plantillas (un pod "Running" escupiendo
# errores). Complementa al Isolation Forest, que es ciego a la severidad.
_SEVERITY_MIN_ERRORS = 5    # mínimo de logs de error para considerarlo
_SEVERITY_LOW = 0.10        # ratio de error donde empieza a puntuar
_SEVERITY_HIGH = 0.40       # ratio de error donde satura a 1.0


def severity_score(window: WindowData) -> float:
    """Sub-score [0,1] por proporción anormal de logs de error en la ventana."""
    if window.error_count < _SEVERITY_MIN_ERRORS:
        return 0.0
    ratio = window.error_ratio
    if ratio <= _SEVERITY_LOW:
        return 0.0
    return min(1.0, (ratio - _SEVERITY_LOW) / (_SEVERITY_HIGH - _SEVERITY_LOW))


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
    ):
        self.bootstrap_windows = bootstrap_windows
        self.rolling_size = rolling_window_size
        self.retrain_every_n = retrain_every_n
        self.threshold = threshold
        self._if_params = dict(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

        self._model: IsolationForest | None = None
        self._model_version: int = 0
        self._trained_cluster_ids: list[int] = []
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

        # Puntuar la ventana: max(score estadístico del IF, señal de severidad).
        if_score, pca_coord = self._score_one(window)
        sev = severity_score(window)
        score = max(if_score, sev)
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
        ), retrained

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

    def _train(self, windows: list[WindowData]) -> None:
        self._trained_cluster_ids = sorted(self._all_cluster_ids)
        X = self._vectorize(windows, self._trained_cluster_ids)
        model = IsolationForest(**self._if_params)
        model.fit(X)
        self._model = model
        self._model_version += 1

        # PCA 2D sobre el training set para visualizacion
        n_components = min(2, X.shape[0], X.shape[1])
        if n_components == 2:
            self._pca = PCA(n_components=2, random_state=42)
            coords = self._pca.fit_transform(X)
            self._training_coords = [(float(c[0]), float(c[1])) for c in coords]
        else:
            self._pca = None
            self._training_coords = [(0.0, 0.0)] * len(windows)

    def get_training_scatter(self, windows: list[WindowData]) -> list[tuple[float, float]]:
        """Coordenadas PCA de las ventanas de entrenamiento actuales."""
        return list(self._training_coords)

    def _score_one(self, window: WindowData) -> float:
        """
        Puntua UNA ventana usando el feature set fijado en el ultimo entrenamiento.
        Cluster_ids nuevos descubiertos despues del entrenamiento se ignoran
        hasta el proximo reentrenamiento (cuando se incorporan al feature set).
        """
        if self._model is None or not self._trained_cluster_ids:
            return 0.0

        history_list = list(self._history)
        all_windows = history_list + [window]
        # Usar siempre el feature set del modelo actual
        X_all = self._vectorize(all_windows, self._trained_cluster_ids)

        raw_scores = self._model.score_samples(X_all)
        s_min, s_max = raw_scores.min(), raw_scores.max()

        if s_max == s_min:
            return 0.0, (0.0, 0.0)

        raw_new = raw_scores[-1]
        normalized = float(np.clip(1.0 - (raw_new - s_min) / (s_max - s_min), 0.0, 1.0))

        # Proyectar la ventana nueva al espacio PCA del modelo actual
        pca_coord = (0.0, 0.0)
        if self._pca is not None:
            v_new = X_all[-1].reshape(1, -1)
            try:
                proj = self._pca.transform(v_new)[0]
                pca_coord = (float(proj[0]), float(proj[1]))
            except Exception:
                pass

        return normalized, pca_coord
