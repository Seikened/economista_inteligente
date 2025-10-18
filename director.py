import pathlib
from app.modelo_fundamental import graficas, Empresa , empresas_lit, FetchNews
from colorstreak import Logger
import polars as pl


class EmpresasDirectory:
    def __init__(self, empresas_tickers: list[empresas_lit], graficas: bool = False) -> None:
        self.empresas_tickers = empresas_tickers
        self._graficas = graficas
        self._empresas: dict[empresas_lit, Empresa] = {}
        self._init_class()
        
    def _init_class(self):
        self._crear_json_noticias()
        self._cargar_empresas_clasificadas()

    def _crear_json_noticias(self):
        """Crea un archivo JSON con noticias si no existe."""
        
        ruta_actual = pathlib.Path().resolve()
        ruta_json = f"{ruta_actual}/noticias.json"
        if not pathlib.Path(ruta_json).exists():
            Logger.info("No se encontró el archivo de noticias. Creando uno nuevo...")
            news = FetchNews(tickers=self.empresas_tickers, days_back=365*7) # type: ignore
            news.fetch_news()
            news.save_to_json(ruta_json)
            Logger.info("Archivo de noticias creado.")
            return
        Logger.info("Archivo de noticias ya existe.")
        return
        
    def _cargar_empresas_clasificadas(self) -> None:
        """Carga y clasifica las noticias para cada empresa."""
        positivo, neutro, negativo = 0, 0, 0
        for ticker in self.empresas_tickers:
            #Logger.info(f"Cargando datos de {ticker}...")
            empresa_obj = Empresa(ticker)
            empresa_all_info = empresa_obj.all_info # Aqui se cargan y clasifican las noticias
            self._empresas[ticker] = empresa_obj
            
            if self._graficas:
                #Logger.info(f"Generando gráficos para {ticker}...")
                
                positivo_h, neutro_h, negativo_h = self._helper_get_sentimientos(empresa_all_info)
                positivo += positivo_h
                negativo += negativo_h
                neutro += neutro_h
                graficas(positivo_h, negativo_h, neutro_h, ticker)
        # Graficos acumulados
        if self._graficas:
            #Logger.info("Generando gráficos acumulados...")
            graficas(positivo, negativo, neutro, "ACUMULADO")

        Logger.info("Todas las empresas han sido cargadas y clasificadas.")
        return

    def _helper_get_sentimientos(self, empresa) -> tuple[int, int, int]:
        resultados_df = (
            empresa
            .group_by("sentiment")
            .agg(pl.len().alias("total"))
            .sort("sentiment")
            .to_dicts()
        )
        resultados = {row["sentiment"]: row["total"] for row in resultados_df}
        positivo = resultados.get("positive", 0)
        neutro = resultados.get("neutral", 0)
        negativo = resultados.get("negative", 0)
        return positivo, neutro, negativo


    def total_noticia_empresa(self, ticker: empresas_lit) -> int:
        """Retorna el total de noticias clasificadas para una empresa específica."""
        empresa = self._empresas.get(ticker)
        if empresa:
            return empresa.all_info.height
        Logger.warning(f"No se encontró la empresa con ticker {ticker}.")
        return 0
    
    def generar_data_set_global(self) -> pl.DataFrame:
        """Genera un DataFrame global con todas las noticias clasificadas."""
        all_data = pl.DataFrame()
        for empresa in self._empresas.values():
            all_data = pl.concat([all_data, empresa.all_info], how="vertical")
        return all_data
    
    @property
    def total_noticias(self) -> int:
        """Retorna el total de noticias clasificadas."""
        return sum(empresa.all_info.height for empresa in self._empresas.values())

    @property
    def empresas(self):
        empresas: list[tuple[empresas_lit, Empresa]] = [(ticker, empresa) for ticker, empresa in self._empresas.items()]
        return empresas

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

import numpy as np

def ratio_sharpe(rendimientos: np.array) -> float:
    dias_trading = 252
    rf_anual = 0.074 
    rf_diaria = rf_anual / dias_trading
    
    rp = rendimientos.mean().item()
    sigma = rendimientos.std(axis=0).item()

    sharpe_diario = (rp - rf_diaria) / sigma
    sharpe_anual = sharpe_diario * np.sqrt(dias_trading)

    return sharpe_diario, sharpe_anual



directorio = EmpresasDirectory(empresas_tickers, graficas=True)
print(f"Total de noticias clasificadas: {directorio.total_noticias}")

for ticker, empresa in directorio.empresas:
    informacion_empresa = empresa.all_info
    cierres  = empresa.cierres
    rendimiento_simple = empresa.rendimiento_simple
    rendimiento_logaritmico = empresa.rendimiento_log
    print(ticker)
    Logger.info(f"Toda la información  {informacion_empresa}")
    Logger.info(f"Cierres  {cierres}| {len(cierres)}")
    Logger.info(f"Rendimiento Simple  {rendimiento_simple} | {len(rendimiento_simple)} ")
    Logger.info(f"Rendimiento Logaritmico  {rendimiento_logaritmico} | {len(rendimiento_logaritmico)}")
    sharpe_diario, sharpe_anual = ratio_sharpe(rendimiento_logaritmico)
    print(f"Ratio de Sharpe Diario: {sharpe_diario:.4f}, Anual: {sharpe_anual:.4f}")
    input("Presiona Enter para continuar...")

