import polars as pl
import json
from typing import Literal
import numpy as np
from transformers import pipeline
from colorstreak import Logger
import matplotlib.pyplot as plt

# ===============================================================================================




class ClasificadorSentimientos:
    def __init__(self,noticia : dict):
        self.noticia = noticia
        self.clasificador = pipeline(
            task="text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            truncation=True,            # corta seguro a 512 tokens
            max_length=512,
        )
        
        
        self.probabilidad_positiva = 0.0
        self.probabilidad_negativa = 0.0
        self.probabilidad_neutra = 0.0
        self.sentimiento = ""


    def _ordenar_probabilidades(self, resultados):
        puntajes_ordenados = sorted(resultados, key=lambda x: x["score"], reverse=True)
        etiqueta_principal = puntajes_ordenados[0]
        suma_total = sum(p["score"] for p in resultados) or 1.0
        probabilidades = {p["label"]: p["score"]/suma_total for p in resultados}
        return etiqueta_principal["label"], {k: round(v, 4) for k, v in probabilidades.items()}


    def _clasificar_titulo_noticia(self):
        titulo = self.noticia.get("headline", "")
        resultados = self.clasificador(titulo)

        puntajes = resultados[0] 
        etiqueta_principal, probabilidades = self._ordenar_probabilidades(puntajes)

        resultado = {
            "sent": etiqueta_principal,
            "positive": probabilidades.get("positive", 0.0),
            "negative": probabilidades.get("negative", 0.0),
            "neutral": probabilidades.get("neutral", 0.0)
        }
        return resultado


    def _clasificar_resumen_noticia(self):
        resumen = self.noticia.get("summary", "")
        tamaño_noticia = len(resumen)
        
        # Si viene vacío o solo espacios, sal temprano con neutro
        if not resumen.strip():
            return {"sent": "neutral", "positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        chunks = []
        if tamaño_noticia > 512:
            chunks = [resumen[i:i + 512] for i in range(0, len(resumen), 512)]
        else:
            chunks = [resumen]
        
        resultados_lista = [self.clasificador(chunk) for chunk in chunks]
        positivos = []
        negativos = []
        neutros = []
        for resultados in resultados_lista:
            puntajes = resultados[0] 
            etiqueta_principal, probabilidades = self._ordenar_probabilidades(puntajes)
            positivos.append(probabilidades.get("positive", 0.0))
            negativos.append(probabilidades.get("negative", 0.0))
            neutros.append(probabilidades.get("neutral", 0.0))
        
        # Promedio pero por pesos de los chunks
        total_peso = sum(len(chunk) for chunk in chunks)
        pesos = [len(chunk) / total_peso for chunk in chunks]
        positivos = [p * peso for p, peso in zip(positivos, pesos)]
        negativos = [n * peso for n, peso in zip(negativos, pesos)]
        neutros = [ne * peso for ne, peso in zip(neutros, pesos)]
        
        # Promedio simple
        probabilidad_promedio = {
            "positive": round(sum(positivos), 4),
            "negative": round(sum(negativos), 4),
            "neutral": round(sum(neutros), 4)
        }
        
        etiqueta_principal = max(probabilidad_promedio, key=probabilidad_promedio.get)
        
        resultado = {
            "sent": etiqueta_principal,
            "positive": probabilidad_promedio["positive"],
            "negative": probabilidad_promedio["negative"],
            "neutral": probabilidad_promedio["neutral"]
        }
        return resultado
    
    
            
    
    
    def clasificar_noticia(self):
        resultado_titulo = self._clasificar_titulo_noticia()
        resultado_resumen = self._clasificar_resumen_noticia()
        peso_titulo = 0.6
        peso_resumen = 0.4
        self.probabilidad_positiva = round((resultado_titulo["positive"] * peso_titulo + resultado_resumen["positive"] * peso_resumen), 4)
        self.probabilidad_negativa = round((resultado_titulo["negative"] * peso_titulo + resultado_resumen["negative"] * peso_resumen), 4)
        self.probabilidad_neutra = round((resultado_titulo["neutral"] * peso_titulo + resultado_resumen["neutral"] * peso_resumen), 4)

        probabilidades = {
            "positive": self.probabilidad_positiva,
            "negative": self.probabilidad_negativa,
            "neutral": self.probabilidad_neutra
        }
        self.sentimiento = max(probabilidades, key=probabilidades.get)
        probabilidad_mas_alta = probabilidades[self.sentimiento]
        
        return self.sentimiento, probabilidad_mas_alta



    @classmethod
    def predecir(cls, noticia: dict):
        instancia = cls(noticia)
        return instancia.clasificar_noticia()

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
        self._link ='/Users/ferleon/Github/economista_inteligente/noticias.json'
        self._noticias: pl.LazyFrame = self._cargar_noticias(self._link)
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
    
    
    def _listar_noticias(self):
        noticias =(
            self._noticias_filtradas()
            .select("headline","summary")    

        )
        return noticias
    
    
# ============================ METODOS DE USO ============================
    
    @property
    def noticias_tabla(self) -> pl.DataFrame:
        " Devuelve un DataFrame con las noticias filtradas de la empresa."
        return self._noticias_filtradas().collect()
    
    
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
    
    


    def clasificar_noticias(self) -> pl.LazyFrame:
        # Esquema para la salida
        esquema_de_salida = pl.Struct([
            pl.Field("sentiment", pl.Utf8),
            pl.Field("probability", pl.Float64),
        ])

        # Mapeo por fila: usa tu @classmethod predecir para no instanciar manualmente aquí
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
            
        
    



# ================ LISTAR NOTICIAS ================

def graficas(resultados, empresa):
    print("=== Resultados Clasificación ===")
    total_positivos = resultados.get("positive", {}).get("total", 0)
    total_neutros = resultados.get("neutral", {}).get("total", 0)
    total_negativos = resultados.get("negative", {}).get("total", 0)

    Logger.info(" Sentimiento | Total")
    Logger.info("-----------------------")
    Logger.info(f" Positivo   | {total_positivos} {total_positivos / sum([total_positivos, total_neutros, total_negativos]):.2%}")
    Logger.info(f" Neutro     | {total_neutros} {total_neutros / sum([total_positivos, total_neutros, total_negativos]):.2%}")
    Logger.info(f" Negativo   | {total_negativos} {total_negativos / sum([total_positivos, total_neutros, total_negativos]):.2%}")
    Logger.info(f"Total Clasificaciones: {total_positivos + total_neutros + total_negativos}")
    Logger.info("-----")

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
    ax.set_title(f"Clasificación de Noticias — {getattr(empresa, '_empresa', 'empresa')}")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()

    # Fondo transparente y nombre de archivo seguro
    safe_name = str(getattr(empresa, "_empresa", "empresa")).replace("/", "-").strip()
    plt.savefig(f"clasificacion_noticias_{safe_name}.png", bbox_inches="tight", transparent=False)
    plt.close(fig)

def clasificar_noticia(noticia, resultados):
    Logger.info("="*70)
    titulo = noticia["headline"]
    resumen = noticia["summary"]
    Logger.info(f"Título: {titulo}")
    Logger.info(f"Resumen: {resumen}")
    # ================ CLASIFICAR NOTICIAS ================
    clasificador = ClasificadorSentimientos(noticia)
    sentimiento, probabilidad = clasificador.clasificar_noticia()
    resultados[sentimiento] = {
        # Le simamos 1 si ya existe, sino es la primera vez
        "total": resultados.get(sentimiento, {}).get("total", 0) + 1
    }
    Logger.debug(f"Sentimiento: {sentimiento}, Probabilidad: {probabilidad}")



# ================ CARGA DE NOTICIAS ================
empresas_tickers =[
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

def crear_empresa(empresa):
    Logger.info(f"Cargando datos de {empresa}...")
    empresa_obj = Empresa(empresa)
    Logger.debug(f"Datos de {empresa} cargados.")
    return empresa_obj


empresas = [crear_empresa(empresa) for empresa in empresas_tickers]




for empresa in empresas:
    print("=== Listar Noticias ===")
    noticias_clasificadas = empresa.clasificar_noticias()
    print(noticias_clasificadas)

    
    #graficas(empresa)