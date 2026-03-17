import numpy as np
from colorstreak import Logger


class ModeloAR:
    def __init__(self, rendimientos: np.ndarray, orden: int = 5) -> None:
        self._rendimientos = rendimientos
        self._orden = orden
        self._coeficientes: np.ndarray | None = None
        self._intercepto: float = 0.0
        self._ajustar()

    def _ajustar(self) -> None:
        """Ajusta el modelo AR(p) usando mínimos cuadrados ordinarios."""
        y = self._rendimientos
        p = self._orden
        n = len(y)

        if n <= p + 1:
            Logger.warning(f"AR({p}): datos insuficientes ({n} puntos), se necesitan al menos {p + 2}")
            return

        # Construir matriz de diseño: cada fila tiene [y(t-1), y(t-2), ..., y(t-p)]
        X = np.column_stack([y[p - i - 1: n - i - 1] for i in range(p)])
        Y = y[p:]

        # Agregar columna de 1s para intercepto
        X = np.column_stack([np.ones(len(Y)), X])

        # Mínimos cuadrados: beta = (X'X)^-1 X'Y
        try:
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            self._intercepto = beta[0]
            self._coeficientes = beta[1:]
        except np.linalg.LinAlgError:
            Logger.warning("AR: no se pudo resolver el sistema de ecuaciones")

    def predecir(self, pasos: int = 5) -> np.ndarray:
        """Predice los próximos N rendimientos."""
        if self._coeficientes is None:
            return np.array([])

        ultimos = list(self._rendimientos[-self._orden:])
        predicciones = []

        for _ in range(pasos):
            ventana = np.array(ultimos[-self._orden:])[::-1]  # más reciente primero
            pred = self._intercepto + np.dot(self._coeficientes, ventana)
            predicciones.append(float(pred))
            ultimos.append(pred)

        return np.array(predicciones)

    @property
    def resumen(self) -> dict:
        """Resumen del modelo: coeficientes, predicción siguiente, tendencia."""
        if self._coeficientes is None:
            return {"ajustado": False}

        predicciones = self.predecir(pasos=5)
        tendencia = "ALCISTA" if predicciones.mean() > 0 else "BAJISTA"

        return {
            "ajustado": True,
            "orden": self._orden,
            "intercepto": round(float(self._intercepto), 6),
            "coeficientes": [round(float(c), 6) for c in self._coeficientes],
            "prediccion_5_dias": [round(float(p), 6) for p in predicciones],
            "tendencia": tendencia,
        }
