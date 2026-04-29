import polars as pl


def indice_diario(noticias: pl.DataFrame) -> pl.DataFrame:
    """Construye el indice de sentimiento diario: (positivos - negativos) / total."""
    return (
        noticias
        .group_by("date")
        .agg(
            (pl.col("sentiment") == "positive").sum().alias("positivos"),
            (pl.col("sentiment") == "negative").sum().alias("negativos"),
            pl.col("sentiment").count().alias("total_noticias"),
        )
        .with_columns(
            ((pl.col("positivos") - pl.col("negativos")) / pl.col("total_noticias"))
            .alias("indice_sentimiento")
        )
        .sort("date")
    )
