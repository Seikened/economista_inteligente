import polars as pl
import json
from typing import Literal
import numpy as np
from colorstreak import Logger
import matplotlib.pyplot as plt
from .modelo_clasificador import ClasificadorSentimientos
import pathlib



# ===============================================================================================
empresas_lit = Literal[
        "NVDA",  # NVIDIA
        "MSFT",  # Microsoft
        "AAPL",  # Apple
        "GOOGL", # Alphabet (Google)
        "AMZN",  # Amazon
        "META",  # Meta (Facebook)
        "AVGO",  # Broadcom
        "TSLA",  # Tesla
        "TSM",   # Taiwan Semiconductor (TSMC)
        "ORCL",  # Oracle
        "TCEHY", # Tencent
        "NFLX",  # Netflix
        "PLTR",  # Palantir
        "BABA",  # Alibaba
        "ASML",  # ASML Holding
        "SAP",   # SAP SE
        "CSCO",  # Cisco
        "IBM",   # IBM
        "AMD"    # Advanced Micro Devices
    ]

class Empresa():
    def __init__(self, empresa: empresas_lit, tokenizar: bool = False) -> None:
        self._link =f'{pathlib.Path().resolve()}/noticias.json'
        self._noticias: pl.LazyFrame = self._cargar_noticias(self._link)
        self._empresa = empresa
        self._tokenizar = tokenizar
        self._ruta_base = f'{pathlib.Path().resolve()}/data/'

    def _cargar_noticias(self, ruta: str) -> pl.LazyFrame:
        """ Carga las noticias desde un archivo JSON y las convierte en un LazyFrame de Polars."""
        with open(ruta, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 2. Convertir a lista de registros planos
        rows = []
        for ticker, contenido in data.items():
            noticias = contenido["data"]
            for n in noticias:
                n["ticker"] = ticker  # Agregamos la empresa como columna
                rows.append(n)

        noticias = pl.LazyFrame(rows)
        noticias = noticias.select(pl.all().exclude(["source", "url"]))
        noticias = noticias.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))
        return noticias


    def _filtrar_empresa(self, ticker: str) -> pl.LazyFrame:
        """ Filtra las noticias de una empresa específica y las ordena por fecha."""
        empresa_lf = (
            self._noticias
            .filter(pl.col("ticker") == ticker)
            .with_columns(pl.col("date").cast(pl.Date))  # ya viene parseada; esto es idempotente
            .sort("date")  # viejo -> nuevo
        )
        return empresa_lf


    def _tratar_cierre(self, empresa: pl.LazyFrame) -> pl.LazyFrame:
        """ Obtiene los precios de cierre diarios y calcula los rendimientos logarítmicos y simples."""
        empresa_por_dia = (
            empresa
            .group_by("date")
            .agg(
                pl.col("close")
                .last()
                .alias("close"))
            .sort("date")
        )
        empresa_rendimientos = (
            empresa_por_dia
            .with_columns(
                pl.col("close").log().diff().alias("rend_log"),
                pl.col("close").pct_change().alias("rend_simple")
            )
            .drop_nulls(subset=["rend_log", "rend_simple"]) # Elimina filas con valores nulos en rendimientos
        )
        return empresa_rendimientos
    

    def _noticias_filtradas(self) -> pl.LazyFrame:
        empresa_noticias: pl.LazyFrame = self._filtrar_empresa(self._empresa)
        columnas_a_excluir = ["close", "ticker", "rend_log", "rend_simple","performance","label"]
        empresa_noticias = empresa_noticias.select(pl.all().exclude(columnas_a_excluir))
        return empresa_noticias
    
    
    def _listar_noticias(self) -> pl.LazyFrame:
        noticias =(
            self._noticias_filtradas()
            .select("headline","summary","date")    

        )
        return noticias
    
    
    def _clasificar_noticias(self) -> pl.LazyFrame:
        # Esquema para la salida
        esquema_de_salida = pl.Struct([
            pl.Field("sentiment", pl.Utf8),
            pl.Field("probability", pl.Float64),
        ])

        # Mapeo por fila
        def _clasificar_fila(noticia: dict) -> dict:
            titulo = noticia.get("headline")
            resumen = noticia.get("summary")
            sentimiento, probabilidad = ClasificadorSentimientos.predecir({"headline": titulo, "summary": resumen})
            return {"sentiment": sentimiento, "probability": probabilidad}

        noticias_clasificadas = (
            self._listar_noticias()
            .with_columns(
                pl.struct(["headline", "summary"])
                .map_elements(_clasificar_fila, return_dtype=esquema_de_salida)
                .alias("pred")
            )
            .unnest("pred") # Desempaqueta las columnas del struct
        )
        return noticias_clasificadas

    
    def _ordenar_noticias(self) -> pl.LazyFrame:
        """ Mezcla las noticias de una empresa con sus precios de cierre diarios."""
        empresa_noticias: pl.LazyFrame = self._filtrar_empresa(self._empresa)
        precios_rendimientos = self._tratar_cierre(empresa_noticias)
        empresa_noticias_clasificadas = self._clasificar_noticias()

        mezclar_noticias_precios = (
            empresa_noticias
            .join(precios_rendimientos , on="date", how="left")
            .join(empresa_noticias_clasificadas, on=["headline", "summary", "date"], how="left")
            .select(
                pl.all().
                exclude("close_right")
            )
            .drop_nulls(subset=["rend_log", "rend_simple"]) # Elimina filas sin precios de cierre asociados
        )
        return mezclar_noticias_precios
    
    def _crear_parquet(self,lazyframe: pl.LazyFrame) -> None:
        """ Guarda las noticias y precios de cierre diarios de la empresa en un archivo Parquet."""
        
        path = self._ruta_base + "parquets_finanzas"
        ruta = pathlib.Path(f"{path}/noticias_{self._empresa}.parquet")
        dataframe = lazyframe.collect()
        dataframe.write_parquet(ruta) # Guardar como Parquet
        #Logger.info("Archivo Parquet guardado 💾")
        
    def _cargar_parquet(self,) -> pl.LazyFrame:
        """ Carga las noticias y precios de cierre diarios de la empresa desde un archivo Parquet."""
        
        path = self._ruta_base + "parquets_finanzas"
        ruta = pathlib.Path(f"{path}/noticias_{self._empresa}.parquet")
        lazyframe = pl.read_parquet(ruta).lazy() # Cargar como LazyFrame
        #Logger.info("Archivo Parquet cargado 📂")
        return lazyframe

    def _cache_parquet(self, lazyframe: pl.LazyFrame) -> pl.LazyFrame:
        """ Carga el archivo Parquet si existe, de lo contrario lo crea."""
        
        path = self._ruta_base + "parquets_finanzas"
        ruta = pathlib.Path(f"{path}/noticias_{self._empresa}.parquet")
        if ruta.exists():
            #Logger.info(f"Parquet ya existe {ruta}, cargando...")
            return self._cargar_parquet()
        else:
            #Logger.info(f"Parquet no existe {ruta}, creando...")
            self._crear_parquet(lazyframe)
            return lazyframe

# ============================ METODOS DE USO ============================

    @property
    def all_info(self) -> pl.DataFrame:
        """ Devuelve un DataFrame con todas las noticias y precios de cierre diarios de la empresa."""
        all_info_df = self._cache_parquet(self._ordenar_noticias()).collect()
        return all_info_df
    
    
    @property
    def noticias_tabla(self) -> pl.DataFrame:
        " Devuelve un DataFrame con las noticias filtradas de la empresa."
        noticias_df = self._noticias_filtradas().collect()
        return noticias_df
    
    
    @property
    def cierres(self) -> np.ndarray:
        " Devuelve un array numpy con los precios de cierre diarios de la empresa."
        empresa_noticias: pl.LazyFrame = self._filtrar_empresa(self._empresa)
        precios_rendimientos = self._tratar_cierre(empresa_noticias)
        return precios_rendimientos.select("close").collect().to_numpy().flatten()
    
    
    @property
    def rendimiento_simple(self) -> np.ndarray:
        " Devuelve un array numpy con los rendimientos simples diarios de la empresa."
        empresa_noticias: pl.LazyFrame = self._filtrar_empresa(self._empresa)
        precios_rendimientos = self._tratar_cierre(empresa_noticias)
        return precios_rendimientos.select("rend_simple").collect().to_numpy().flatten()
    
    @property
    def rendimiento_log(self) -> np.ndarray:
        " Devuelve un array numpy con los rendimientos logarítmicos diarios de la empresa."
        empresa_noticias: pl.LazyFrame = self._filtrar_empresa(self._empresa)
        precios_rendimientos = self._tratar_cierre(empresa_noticias)
        return precios_rendimientos.select("rend_log").collect().to_numpy().flatten()
    
    



            
        
    



# ================ GRPÁFICAR NOTICIAS ================

def graficas(positivo, negativo, neutro, empresa_ticker: empresas_lit | str) -> None:


    total_positivos = positivo
    total_neutros = neutro
    total_negativos = negativo

    # --- Plot bonito y minimal ---
    labels = ["Positivo", "Neutro", "Negativo"]
    values = [total_positivos, total_neutros, total_negativos]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=144)

    # Paleta “limpia”: verde, azul, rojo (tono moderno)
    palette = ["#22c55e", "#3b82f6", "#ef4444"]

    bars = ax.bar(labels, values, color=palette)

    # Anotar cada barra con conteo y porcentaje
    total = sum(values)
    for rect, val in zip(bars, values):
        pct = f" ({val/total:.1%})" if total > 0 else ""
        ax.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                f"{val:,}{pct}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Total")
    ax.set_title(f"Clasificación de Noticias — {empresa_ticker}")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()

    # Fondo transparente y nombre de archivo seguro
    safe_name = str(empresa_ticker).replace("/", "-").strip()
    ruta = f"{pathlib.Path().resolve()}/data/graficas"
    plt.savefig(f"{ruta}/clasificacion_noticias_{safe_name}.png", bbox_inches="tight", transparent=False)
    plt.close(fig)




if __name__ == "__main__":
    # ================ CARGA DE NOTICIAS ================
    empresas_tickers: list[empresas_lit] = [
            "NVDA",  # NVIDIA
            # "MSFT",  # Microsoft
            # "AAPL",  # Apple
            # "GOOGL", # Alphabet (Google)
            # "AMZN",  # Amazon
            # "META",  # Meta (Facebook)
            # "AVGO",  # Broadcom
            # "TSLA",  # Tesla
            # "TSM",   # Taiwan Semiconductor (TSMC)
            # "ORCL",  # Oracle
            # "TCEHY", # Tencent
            # "NFLX",  # Netflix
            # "PLTR",  # Palantir
            # "BABA",  # Alibaba
            # "ASML",  # ASML Holding
            # "SAP",   # SAP SE
            # "CSCO",  # Cisco
            # "IBM",   # IBM
            # "AMD"    # Advanced Micro Devices
        ]


    def clasificar_empresa(empresa: empresas_lit) -> tuple[empresas_lit, pl.DataFrame]:
        Logger.info(f"Cargando datos de {empresa}...")
        empresa_obj = Empresa(empresa)
        df = empresa_obj.all_info
        Logger.info(f"Noticias clasificadas de {empresa}...")
        # empresa_obj.crear_parquet(df)
        # Logger.info("Parquet creado.")
        return empresa, df

    from time import perf_counter
    start_time = perf_counter()
    empresas_clasificadas = [clasificar_empresa(empresa) for empresa in empresas_tickers]
    end_time = perf_counter()
    Logger.debug(f"TIEMPO DE CLASIFICACIÓN: {end_time - start_time:.2f} SEGUNDOS")

    total_noticias = 0
    positivo_acumulado = 0.0
    negativo_acumulado = 0.0
    neutro_acumulado = 0.0
    for ticker, noticias_clasificadas in empresas_clasificadas:
        print("=== Listar Noticias ===")
        print(f"Empresa: {ticker}")
        print(noticias_clasificadas)
        
        total_noticias += noticias_clasificadas.height
        
        resultados_df = (
            noticias_clasificadas
            .group_by("sentiment")
            .agg(pl.count().alias("total"))
            .sort("sentiment")
            .to_dicts()
        )
        

        resultados = {row["sentiment"]: row["total"] for row in resultados_df}
        positivo = resultados.get("positive", 0)
        neutro = resultados.get("neutral", 0)
        negativo = resultados.get("negative", 0)

        graficas(positivo, negativo, neutro, ticker)
        positivo_acumulado += positivo
        neutro_acumulado += neutro
        negativo_acumulado += negativo

    graficas(positivo_acumulado, negativo_acumulado, neutro_acumulado, "ACUMULADO")
    Logger.info(f"Total Positivos: {positivo_acumulado}")
    Logger.info(f"Total Neutros: {neutro_acumulado}")
    Logger.info(f"Total Negativos: {negativo_acumulado}")
    Logger.info(f"Total Noticias Clasificadas: {total_noticias}")
    Logger.info(f"Tiempo por noticia: {total_noticias / (end_time - start_time):.2f} noticias/segundo")
    
        