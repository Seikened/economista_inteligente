import polars as pl
import numpy as np
from app.modelo_matematico import ratio_sharpe


def evaluar(df: pl.DataFrame, columna_senal: str) -> dict:
    """Evalua una estrategia: la senal del dia t opera el rendimiento del dia t+1."""
    datos = (
        df
        .with_columns(pl.col(columna_senal).shift(1).alias("posicion"))
        .drop_nulls(subset=["posicion", "rend_simple"])
        .with_columns((pl.col("posicion") * pl.col("rend_simple")).alias("rend_estrategia"))
    )

    rendimientos = datos.select("rend_estrategia").to_numpy().flatten()
    if rendimientos.size < 2:
        return {"rend_acumulado": 0.0, "sharpe": 0.0, "precision": 0.0, "volatilidad": 0.0, "operaciones": 0}

    rend_acumulado = float(np.prod(1 + rendimientos) - 1) * 100
    _, sharpe_anual = ratio_sharpe(rendimientos)
    volatilidad_anual = float(np.std(rendimientos, ddof=1) * np.sqrt(252) * 100)

    activos = datos.filter(pl.col("posicion") != 0)
    if activos.height > 0:
        aciertos = activos.filter(
            ((pl.col("posicion") > 0) & (pl.col("rend_simple") > 0))
            | ((pl.col("posicion") < 0) & (pl.col("rend_simple") < 0))
        ).height
        precision = aciertos / activos.height * 100
    else:
        precision = 0.0

    return {
        "rend_acumulado": rend_acumulado,
        "sharpe": float(sharpe_anual),
        "precision": precision,
        "volatilidad": volatilidad_anual,
        "operaciones": activos.height,
    }
