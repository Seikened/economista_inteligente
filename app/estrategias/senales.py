import polars as pl

COMPRA = 1
VENTA = -1
HOLD = 0


def senal_tecnica(df: pl.DataFrame) -> pl.DataFrame:
    """COMPRA si SMA20 > SMA50 (golden), VENTA si SMA20 < SMA50 (death)."""
    return df.with_columns(
        pl.when(pl.col("sma_20") > pl.col("sma_50")).then(COMPRA)
        .when(pl.col("sma_20") < pl.col("sma_50")).then(VENTA)
        .otherwise(HOLD)
        .cast(pl.Int8)
        .alias("senal_tecnica")
    )


def senal_sentimiento(df: pl.DataFrame, umbral: float = 0.2) -> pl.DataFrame:
    """COMPRA si indice > umbral, VENTA si indice < -umbral."""
    return df.with_columns(
        pl.when(pl.col("indice_sentimiento") > umbral).then(COMPRA)
        .when(pl.col("indice_sentimiento") < -umbral).then(VENTA)
        .otherwise(HOLD)
        .cast(pl.Int8)
        .alias("senal_sentimiento")
    )


def senal_hibrida(df: pl.DataFrame) -> pl.DataFrame:
    """Tecnica + sentimiento coinciden, filtrado por Bollinger (no comprar en sobrecompra ni vender en sobreventa)."""
    tecnica = pl.col("senal_tecnica")
    sentimiento = pl.col("senal_sentimiento")
    no_sobrecompra = pl.col("close") < pl.col("bollinger_sup")
    no_sobreventa = pl.col("close") > pl.col("bollinger_inf")
    return df.with_columns(
        pl.when((tecnica == COMPRA) & (sentimiento == COMPRA) & no_sobrecompra).then(COMPRA)
        .when((tecnica == VENTA) & (sentimiento == VENTA) & no_sobreventa).then(VENTA)
        .otherwise(HOLD)
        .cast(pl.Int8)
        .alias("senal_hibrida")
    )


def mantener_posicion(df: pl.DataFrame, columna: str) -> pl.DataFrame:
    """Mantiene la ultima posicion activa: HOLD hereda la senal previa."""
    return df.with_columns(
        pl.when(pl.col(columna) == HOLD).then(None).otherwise(pl.col(columna))
        .forward_fill()
        .fill_null(HOLD)
        .cast(pl.Int8)
        .alias(columna)
    )
