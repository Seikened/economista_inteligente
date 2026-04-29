import pathlib
import polars as pl
import yfinance as yf

from app.modelo_fundamental import Empresa
from .indicadores import agregar_indicadores
from .sentimiento_diario import indice_diario
from .senales import senal_tecnica, senal_sentimiento, senal_hibrida, mantener_posicion
from .backtest import evaluar


ESTRATEGIAS = ("Tecnica", "Sentimiento", "Hibrida")
_COLUMNAS_SENAL = {
    "Tecnica": "senal_tecnica",
    "Sentimiento": "senal_sentimiento",
    "Hibrida": "senal_hibrida",
}
_CACHE_PRECIOS = pathlib.Path("data/parquets_finanzas")


def _serie_diaria(ticker: str, periodo: str = "7y") -> pl.DataFrame:
    """Descarga (o lee de cache) la serie diaria completa de cierres."""
    cache = _CACHE_PRECIOS / f"precios_{ticker}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)

    data = yf.download(ticker, period=periodo, progress=False, auto_adjust=True)
    if data is None or data.empty:
        return pl.DataFrame(schema={"date": pl.Date, "close": pl.Float64, "rend_simple": pl.Float64})

    if hasattr(data.columns, "get_level_values"):
        data.columns = data.columns.get_level_values(0)
    serie = data[["Close"]].reset_index().rename(columns={"Date": "date", "Close": "close"})

    df = (
        pl.from_pandas(serie)
        .with_columns(pl.col("date").cast(pl.Date), pl.col("close").cast(pl.Float64))
        .with_columns(pl.col("close").pct_change().alias("rend_simple"))
        .drop_nulls()
        .sort("date")
    )

    _CACHE_PRECIOS.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    return df


def construir_dataset(empresa: Empresa) -> pl.DataFrame:
    """Une serie diaria de precios + sentimiento + indicadores + senales."""
    serie_precios = _serie_diaria(empresa._empresa)
    sentimiento = indice_diario(empresa.all_info)

    base = (
        serie_precios
        .join(sentimiento.select("date", "indice_sentimiento"), on="date", how="left")
        .with_columns(pl.col("indice_sentimiento").fill_null(0))
    )

    base = agregar_indicadores(base)
    base = senal_tecnica(base)
    base = senal_sentimiento(base)
    base = mantener_posicion(base, "senal_sentimiento")
    base = senal_hibrida(base)
    base = mantener_posicion(base, "senal_hibrida")
    return base.drop_nulls(subset=["sma_50"])


def evaluar_empresa(empresa: Empresa) -> dict[str, dict]:
    """Devuelve un dict {estrategia: metricas} para una empresa."""
    df = construir_dataset(empresa)
    return {
        nombre: evaluar(df, _COLUMNAS_SENAL[nombre])
        for nombre in ESTRATEGIAS
    }


def promediar(resultados: dict[str, dict[str, dict]]) -> dict[str, dict]:
    """Promedia las metricas de todas las empresas por estrategia."""
    promedios: dict[str, dict] = {}
    for nombre in ESTRATEGIAS:
        metricas = [r[nombre] for r in resultados.values()]
        n = len(metricas) or 1
        promedios[nombre] = {
            "rend_acumulado": sum(m["rend_acumulado"] for m in metricas) / n,
            "sharpe": sum(m["sharpe"] for m in metricas) / n,
            "precision": sum(m["precision"] for m in metricas) / n,
            "volatilidad": sum(m["volatilidad"] for m in metricas) / n,
            "operaciones": sum(m["operaciones"] for m in metricas),
        }
    return promedios
