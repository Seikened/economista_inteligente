import polars as pl
import numpy as np
from colorstreak import Logger


class PromediosMoviles:
    def __init__(self, cierres: np.ndarray, ventana_corta: int = 20, ventana_larga: int = 50) -> None:
        self._cierres = cierres
        self._ventana_corta = ventana_corta
        self._ventana_larga = ventana_larga
        self._df = self._construir()

    def _construir(self) -> pl.DataFrame:
        """Construye el DataFrame con SMA, EMA y señales de cruce."""
        df = pl.DataFrame({"close": self._cierres})
        df = df.with_row_index("t")

        df = df.with_columns(
            pl.col("close").rolling_mean(window_size=self._ventana_corta).alias(f"sma_{self._ventana_corta}"),
            pl.col("close").rolling_mean(window_size=self._ventana_larga).alias(f"sma_{self._ventana_larga}"),
            pl.col("close").ewm_mean(span=self._ventana_corta).alias(f"ema_{self._ventana_corta}"),
            pl.col("close").ewm_mean(span=self._ventana_larga).alias(f"ema_{self._ventana_larga}"),
        )

        col_sma_corta = f"sma_{self._ventana_corta}"
        col_sma_larga = f"sma_{self._ventana_larga}"

        # Señal de cruce: 1 = golden cross (compra), -1 = death cross (venta), 0 = sin cambio
        df = df.with_columns(
            (pl.col(col_sma_corta) - pl.col(col_sma_larga)).sign().alias("_pos"),
        ).with_columns(
            (pl.col("_pos") - pl.col("_pos").shift(1)).alias("senal_cruce"),
        ).drop("_pos")

        # senal_cruce: 2 = compra, -2 = venta, 0 = sin cambio
        df = df.with_columns(
            pl.when(pl.col("senal_cruce") == 2).then(pl.lit("COMPRA"))
            .when(pl.col("senal_cruce") == -2).then(pl.lit("VENTA"))
            .otherwise(pl.lit("HOLD"))
            .alias("senal")
        ).drop("senal_cruce")

        return df.drop_nulls()

    @property
    def vacio(self) -> bool:
        return self._df.height == 0

    @property
    def tabla(self) -> pl.DataFrame:
        return self._df

    @property
    def senales(self) -> pl.DataFrame:
        """Solo las filas donde hay señal de compra o venta."""
        if self.vacio:
            return self._df
        return self._df.filter(pl.col("senal") != "HOLD")

    @property
    def resumen(self) -> dict:
        """Resumen rápido: total señales, última señal, precio actual vs SMAs."""
        if self.vacio:
            return {"datos_insuficientes": True}

        senales = self.senales
        ultima = senales.tail(1).to_dicts()
        ultimo_cierre = self._cierres[-1]
        sma_corta_val = self._df[f"sma_{self._ventana_corta}"][-1]
        sma_larga_val = self._df[f"sma_{self._ventana_larga}"][-1]

        return {
            "total_senales": senales.height,
            "ultima_senal": ultima[0] if ultima else None,
            "precio_actual": float(ultimo_cierre),
            f"sma_{self._ventana_corta}": float(sma_corta_val),
            f"sma_{self._ventana_larga}": float(sma_larga_val),
            "precio_sobre_sma_corta": float(ultimo_cierre) > float(sma_corta_val),
            "precio_sobre_sma_larga": float(ultimo_cierre) > float(sma_larga_val),
        }
