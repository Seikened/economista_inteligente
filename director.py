from app.modelo_fundamental import graficas, Empresa , empresas_lit, FetchNews
from colorstreak import Logger
import polars as pl
from time import perf_counter



# ================ CARGA DE NOTICIAS ================
empresas_tickers: list[empresas_lit] = [
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


# desde hace 7 años
total_tiempo = perf_counter()
news = FetchNews(tickers=empresas_tickers, days_back=365*7)
news.fetch_news()
news.save_to_json("noticias.json")


def clasificar_empresa(empresa: empresas_lit) -> tuple[empresas_lit, pl.DataFrame]:
    Logger.info(f"Cargando datos de {empresa}...")
    empresa_obj = Empresa(empresa)
    df = empresa_obj.all_info
    Logger.info(f"Noticias clasificadas de {empresa}...")
    # empresa_obj.crear_parquet(df)
    # Logger.info("Parquet creado.")
    return empresa, df


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
total_tiempo_fin = perf_counter()
Logger.info(f"Tiempo total de ejecución: {total_tiempo_fin - total_tiempo:.2f} segundos")
Logger.info(f"Total Positivos: {positivo_acumulado}")
Logger.info(f"Total Neutros: {neutro_acumulado}")
Logger.info(f"Total Negativos: {negativo_acumulado}")
Logger.info(f"Total Noticias Clasificadas: {total_noticias}")
Logger.info(f"Tiempo por noticia: {total_noticias / (end_time - start_time):.2f} noticias/segundo")

    