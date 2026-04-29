import polars as pl


def agregar_indicadores(precios: pl.DataFrame) -> pl.DataFrame:
    """Agrega SMA 20, SMA 50 y bandas de Bollinger sobre la columna `close`."""
    return (
        precios
        .with_columns(
            pl.col("close").rolling_mean(20).alias("sma_20"),
            pl.col("close").rolling_mean(50).alias("sma_50"),
            pl.col("close").rolling_std(20).alias("desv_20"),
        )
        .with_columns(
            (pl.col("sma_20") + 2 * pl.col("desv_20")).alias("bollinger_sup"),
            (pl.col("sma_20") - 2 * pl.col("desv_20")).alias("bollinger_inf"),
        )
    )
