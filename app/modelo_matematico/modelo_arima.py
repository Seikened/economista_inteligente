import numpy as np
import warnings
from statsmodels.tsa.arima.model import ARIMA
from colorstreak import Logger


class ModeloARIMA:
    """ARIMA sobre precios crudos. Selecciona el mejor orden (p,d,q) por AIC."""

    # Grilla de búsqueda: (p, d, q)
    _ORDENES = [
        (p, d, q)
        for p in range(0, 4)
        for d in range(1, 3)   # d>=1 porque precios no son estacionarios
        for q in range(0, 3)
    ]

    def __init__(self, precios: np.ndarray, pasos: int = 5) -> None:
        self._precios = precios
        self._pasos = pasos
        self._orden: tuple[int, int, int] | None = None
        self._aic: float | None = None
        self._modelo = None
        self._ajustar()

    def _ajustar(self) -> None:
        """Selecciona el mejor (p,d,q) por AIC y ajusta el modelo."""
        if len(self._precios) < 30:
            Logger.warning("ARIMA: datos insuficientes (se necesitan >=30 precios)")
            return

        mejor_aic = np.inf
        mejor_orden = (1, 1, 0)
        mejor_resultado = None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for orden in self._ORDENES:
                try:
                    resultado = ARIMA(self._precios, order=orden).fit()
                    if resultado.aic < mejor_aic:
                        mejor_aic = resultado.aic
                        mejor_orden = orden
                        mejor_resultado = resultado
                except Exception:
                    continue

        if mejor_resultado is None:
            Logger.warning("ARIMA: no se pudo ajustar ningún modelo")
            return

        self._orden = mejor_orden
        self._aic = mejor_aic
        self._modelo = mejor_resultado

    def predecir(self) -> np.ndarray:
        """Predice los próximos N precios."""
        if self._modelo is None:
            return np.array([])
        forecast = self._modelo.forecast(steps=self._pasos)
        return np.array(forecast)

    @property
    def resumen(self) -> dict:
        if self._modelo is None:
            return {"ajustado": False}

        predicciones = self.predecir()
        ultimo_precio = float(self._precios[-1])
        rendimiento_esperado = float((predicciones[-1] - ultimo_precio) / ultimo_precio)
        tendencia = "ALCISTA" if rendimiento_esperado > 0 else "BAJISTA"

        return {
            "ajustado": True,
            "orden": f"ARIMA{self._orden}",
            "aic": round(float(self._aic), 2),
            "precio_actual": round(ultimo_precio, 2),
            "predicciones": [round(float(p), 2) for p in predicciones],
            "rendimiento_esperado": round(rendimiento_esperado * 100, 4),
            "tendencia": tendencia,
        }
