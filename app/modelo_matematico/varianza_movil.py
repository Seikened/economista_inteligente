import polars as pl
import numpy as np


class VarianzaMovil:
    def __init__(self, cierres: np.ndarray, ventana: int = 20, num_desviaciones: float = 2.0) -> None:
        self._cierres = cierres
        self._ventana = ventana
        self._num_desv = num_desviaciones
        self._df = self._construir()

    def _construir(self) -> pl.DataFrame:
        """Construye Bollinger Bands y métricas de volatilidad."""
        df = pl.DataFrame({"close": self._cierres})
        df = df.with_row_index("t")

        df = df.with_columns(
            pl.col("close").rolling_mean(window_size=self._ventana).alias("sma"),
            pl.col("close").rolling_std(window_size=self._ventana).alias("std"),
            pl.col("close").pct_change().alias("rendimiento"),
        )

        # Bollinger Bands
        df = df.with_columns(
            (pl.col("sma") + self._num_desv * pl.col("std")).alias("banda_superior"),
            (pl.col("sma") - self._num_desv * pl.col("std")).alias("banda_inferior"),
        )

        # Posición del precio dentro de las bandas: 0 = banda inferior, 1 = banda superior
        df = df.with_columns(
            ((pl.col("close") - pl.col("banda_inferior")) /
             (pl.col("banda_superior") - pl.col("banda_inferior")))
            .alias("posicion_bollinger"),
        )

        # Volatilidad rolling del rendimiento
        df = df.with_columns(
            pl.col("rendimiento").rolling_std(window_size=self._ventana).alias("volatilidad"),
        )

        return df.drop_nulls()

    @property
    def vacio(self) -> bool:
        return self._df.height == 0

    @property
    def tabla(self) -> pl.DataFrame:
        return self._df

    @property
    def resumen(self) -> dict:
        """Resumen: volatilidad actual, posición en bandas, si está sobrecomprado/sobrevendido."""
        if self.vacio:
            return {"datos_insuficientes": True}

        ultimo = self._df.tail(1).to_dicts()[0]
        return {
            "volatilidad_actual": ultimo["volatilidad"],
            "posicion_bollinger": ultimo["posicion_bollinger"],
            "precio": ultimo["close"],
            "banda_superior": ultimo["banda_superior"],
            "banda_inferior": ultimo["banda_inferior"],
            "sobrecomprado": ultimo["posicion_bollinger"] > 1.0,
            "sobrevendido": ultimo["posicion_bollinger"] < 0.0,
        }
