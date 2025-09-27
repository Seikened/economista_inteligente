import polars as pl
import json
from typing import Literal
import numpy as np

from .rating_model import ClasificadorSentimientos


empresas = Literal["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "UNH"]


class Empresa():
    def __init__(self,link: str, empresa: empresas, tokenizar: bool = False) -> None:
        self._link = link
        self._noticias: pl.LazyFrame = self._cargar_noticias(link)
        self._empresa = empresa
        self._tokenizar = tokenizar

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
    
    
    def _ordenar_noticias(self):
        """ Mezcla las noticias de una empresa con sus precios de cierre diarios."""
        empresa_noticias: pl.LazyFrame = self._filtrar_empresa(self._empresa)
        precios_rendimientos = self._tratar_cierre(empresa_noticias)

        mezclar_noticias_precios = (
            empresa_noticias
            .join(precios_rendimientos, on="date", how="left")
            .select(
                pl.all().
                exclude("close_right")
            )
            .drop_nulls(subset=["rend_log", "rend_simple"]) # Elimina filas sin precios de cierre asociados
        )
        return mezclar_noticias_precios
    

    @property
    def all_info(self) -> pl.DataFrame:
        """ Devuelve un DataFrame con todas las noticias y precios de cierre diarios de la empresa."""
        return self._ordenar_noticias().collect()
        
        
    
    def _noticias_filtradas(self):
        empresa_noticias: pl.LazyFrame = self._ordenar_noticias()
        columnas_a_excluir = ["close", "ticker", "rend_log", "rend_simple","performance","label"]
        empresa_noticias = empresa_noticias.select(pl.all().exclude(columnas_a_excluir))
        return empresa_noticias
    
    @property
    def noticias_tabla(self) -> pl.DataFrame:
        return self._noticias_filtradas().collect()
    
    def listar_noticias(self):
        noticias =(
            self._noticias_filtradas()
            .select("headline","summary")
        )
        return noticias.collect().to_dicts()
    
    
    
    @property
    def cierres(self) -> np.ndarray:
        """ Devuelve un array numpy con los precios de cierre diarios de la empresa."""
        empresa_noticias: pl.LazyFrame = self._filtrar_empresa(self._empresa)
        precios_rendimientos = self._tratar_cierre(empresa_noticias)
        return precios_rendimientos.select("close").collect().to_numpy().flatten()
    
    
    @property
    def rendimiento_simple(self) -> np.ndarray:
        empresa_noticias: pl.LazyFrame = self._filtrar_empresa(self._empresa)
        precios_rendimientos = self._tratar_cierre(empresa_noticias)
        return precios_rendimientos.select("rend_simple").collect().to_numpy().flatten()
    
    @property
    def rendimiento_log(self) -> np.ndarray:
        empresa_noticias: pl.LazyFrame = self._filtrar_empresa(self._empresa)
        precios_rendimientos = self._tratar_cierre(empresa_noticias)
        return precios_rendimientos.select("rend_log").collect().to_numpy().flatten()
    




# ================ CARGA DE DATOS ================
ruta = '/Users/ferleon/Github/economista_inteligente/noticias.json'
apple = Empresa(ruta, "AAPL", tokenizar=True)
    
# ================ INFO ================
info = apple.all_info
print(info)



# ================ NOTICIAS TABLA ================
noticias = apple.noticias_tabla
print(noticias)


# ================ CIERRES ================
cierres = apple.cierres
for cierre in cierres[:5]:
    print(f"Cierres: {cierre}")


# ================ RENDIMIENTOS SIMPLES ================
rendimiento_simple = apple.rendimiento_simple
for rendimiento in rendimiento_simple[:5]:
    print(f"Rendimiento: {rendimiento}")


# ================ RENDIMIENTO LOGARITMICO ================
rendimiento_log = apple.rendimiento_log
for rendimiento in rendimiento_log[:5]:
    print(f"Rendimiento log: {rendimiento}")



# ================ LISTAR NOTICIAS ================
print("=== Listar Noticias ===")
lista_noticias = apple.listar_noticias()
for noticia in lista_noticias[:3]:
    titulo = noticia["headline"]
    resumen = noticia["summary"]
    print(f"Título: {titulo}")
    print(f"Resumen: {resumen}")
    print("-----")
# ================ CLASIFICAR NOTICIAS ================

    clasificador = ClasificadorSentimientos(noticia)
    sentimiento, probabilidad = clasificador.clasificar_noticia()
    print(f"Sentimiento: {sentimiento}, Probabilidad: {probabilidad}")
