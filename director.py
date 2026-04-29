import pathlib
from app.modelo_fundamental import graficas, Empresa, empresas_lit, FetchNews
from app.modelo_matematico import ratio_sharpe, volatilidad, sortino, ModeloARIMA
from app.estrategias import ESTRATEGIAS, evaluar_empresa, promediar
from colorstreak import Logger
import polars as pl
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns

console = Console()


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

    def _cargar_empresas_clasificadas(self) -> None:
        """Carga y clasifica las noticias para cada empresa."""
        positivo, neutro, negativo = 0, 0, 0
        for ticker in self.empresas_tickers:
            empresa_obj = Empresa(ticker)
            empresa_all_info = empresa_obj.all_info
            self._empresas[ticker] = empresa_obj

            if self._graficas:
                positivo_h, neutro_h, negativo_h = self._helper_get_sentimientos(empresa_all_info)
                positivo += positivo_h
                negativo += negativo_h
                neutro += neutro_h
                graficas(positivo_h, negativo_h, neutro_h, ticker)

        if self._graficas:
            graficas(positivo, negativo, neutro, "ACUMULADO")

        Logger.info("Todas las empresas han sido cargadas y clasificadas.")

    def _helper_get_sentimientos(self, empresa) -> tuple[int, int, int]:
        resultados_df = (
            empresa
            .group_by("sentiment")
            .agg(pl.len().alias("total"))
            .sort("sentiment")
            .to_dicts()
        )
        resultados = {row["sentiment"]: row["total"] for row in resultados_df}
        return resultados.get("positive", 0), resultados.get("neutral", 0), resultados.get("negative", 0)

    def total_noticia_empresa(self, ticker: empresas_lit) -> int:
        empresa = self._empresas.get(ticker)
        if empresa:
            return empresa.all_info.height
        Logger.warning(f"No se encontró la empresa con ticker {ticker}.")
        return 0

    def generar_data_set_global(self) -> pl.DataFrame:
        all_data = pl.DataFrame()
        for empresa in self._empresas.values():
            all_data = pl.concat([all_data, empresa.all_info], how="vertical")
        return all_data

    @property
    def total_noticias(self) -> int:
        return sum(empresa.all_info.height for empresa in self._empresas.values())

    @property
    def empresas(self):
        return [(ticker, empresa) for ticker, empresa in self._empresas.items()]


# ================ HELPERS RICH ================

def _color_tendencia(tendencia: str) -> str:
    return "[bold green]ALCISTA[/]" if tendencia == "ALCISTA" else "[bold red]BAJISTA[/]"


def _imprimir_empresa(ticker: str, arima, num_noticias: int,
                      sharpe_d, sharpe_a, volatil, volatil_a, sortino_d, sortino_a) -> None:

    # ── Ratios Financieros ──
    tabla_rf = Table(show_header=False, box=None, padding=(0, 1))
    tabla_rf.add_column(style="dim")
    tabla_rf.add_column()
    tabla_rf.add_row("Sharpe diario",      f"{sharpe_d:.4f}")
    tabla_rf.add_row("Sharpe anual",       f"{sharpe_a:.4f}")
    tabla_rf.add_row("Volatilidad diaria", f"{volatil:.4f}")
    tabla_rf.add_row("Volatilidad anual",  f"{volatil_a:.4f}")
    tabla_rf.add_row("Sortino diario",     f"{sortino_d:.4f}")
    tabla_rf.add_row("Sortino anual",      f"{sortino_a:.4f}")
    panel_rf = Panel(tabla_rf, title="Ratios Financieros", border_style="yellow")

    # ── ARIMA ──
    ar_r = arima.resumen
    if not ar_r.get("ajustado"):
        panel_ar = Panel("[dim]No se pudo ajustar[/]", title="ARIMA", border_style="yellow")
    else:
        preds = ar_r["predicciones"]
        preds_txt = "  ".join(
            f"[green]${p:,.2f}[/]" if p >= ar_r["precio_actual"] else f"[red]${p:,.2f}[/]"
            for p in preds
        )
        rend = ar_r["rendimiento_esperado"]
        rend_txt = f"[green]+{rend:.2f}%[/]" if rend >= 0 else f"[red]{rend:.2f}%[/]"

        tabla_ar = Table(show_header=False, box=None, padding=(0, 1))
        tabla_ar.add_column(style="dim")
        tabla_ar.add_column()
        tabla_ar.add_row("Modelo",              ar_r["orden"])
        tabla_ar.add_row("AIC",                 f"{ar_r['aic']:,.2f}")
        tabla_ar.add_row("Precio actual",        f"${ar_r['precio_actual']:,.2f}")
        tabla_ar.add_row(f"Pred. {len(preds)} dias", preds_txt)
        tabla_ar.add_row("Rendimiento esperado", rend_txt)
        tabla_ar.add_row("Tendencia",            _color_tendencia(ar_r["tendencia"]))
        panel_ar = Panel(tabla_ar, title="ARIMA", border_style="cyan")

    console.print()
    console.rule(f"[bold yellow]{ticker}[/]  —  {num_noticias:,} noticias", style="yellow")
    console.print(Columns([panel_rf, panel_ar], equal=True, expand=True))


# ================ ESTRATEGIAS (VERSION FINAL) ================

def _color_metrica(valor: float) -> str:
    return f"[green]{valor:+.2f}[/]" if valor >= 0 else f"[red]{valor:+.2f}[/]"


def _tabla_estrategias(titulo: str, metricas_por_estrategia: dict, estilo: str) -> Table:
    tabla = Table(title=titulo, border_style=estilo, title_style=f"bold {estilo}")
    tabla.add_column("Estrategia", style="bold")
    tabla.add_column("Rend. acum. %", justify="right")
    tabla.add_column("Sharpe anual", justify="right")
    tabla.add_column("Volatilidad %", justify="right")
    tabla.add_column("Precision %", justify="right")
    tabla.add_column("Operaciones", justify="right")
    for nombre in ESTRATEGIAS:
        m = metricas_por_estrategia[nombre]
        tabla.add_row(
            nombre,
            _color_metrica(m["rend_acumulado"]),
            _color_metrica(m["sharpe"]),
            f"{m['volatilidad']:.2f}",
            f"{m['precision']:.1f}",
            f"{m['operaciones']:,}",
        )
    return tabla


def _correr_estrategias(directorio: "EmpresasDirectory") -> None:
    console.print()
    console.rule("[bold magenta]ANALISIS DE ESTRATEGIAS — VERSION FINAL[/]", style="magenta")
    console.print()

    resultados: dict[str, dict] = {}
    for ticker, empresa in directorio.empresas:
        metricas = evaluar_empresa(empresa)
        resultados[ticker] = metricas
        console.print(_tabla_estrategias(f"{ticker}", metricas, "magenta"))
        console.print()

    promedios = promediar(resultados)
    console.rule("[bold cyan]RESUMEN GLOBAL[/]", style="cyan")
    console.print(_tabla_estrategias("Promedio sobre todas las empresas", promedios, "cyan"))
    console.print()


# ================ MAIN ================

empresas_tickers: list[empresas_lit] = [
    "NVDA",  "MSFT",  "AAPL",  "GOOGL", "AMZN",
    "META",  "AVGO",  "TSLA",  "TSM",   "ORCL",
    "TCEHY", "NFLX",  "PLTR",  "BABA",  "ASML",
    "SAP",   "CSCO",  "IBM",   "AMD",
]

directorio = EmpresasDirectory(empresas_tickers, graficas=True)

console.print()
console.rule("[bold cyan]ECONOMISTA INTELIGENTE[/]", style="cyan")
console.print(f"  Total de noticias clasificadas: [bold]{directorio.total_noticias:,}[/]")
console.print(f"  Empresas analizadas: [bold]{len(empresas_tickers)}[/]")
console.rule(style="cyan")
console.print()

for ticker, empresa in directorio.empresas:
    cierres           = empresa.cierres
    rendimiento_simple = empresa.rendimiento_simple

    sharpe_d,  sharpe_a  = ratio_sharpe(rendimiento_simple)
    volatil,   volatil_a = volatilidad(rendimiento_simple)
    sortino_d, sortino_a = sortino(rendimiento_simple)

    arima = ModeloARIMA(cierres, pasos=5)

    _imprimir_empresa(ticker, arima, empresa.all_info.height,
                      sharpe_d, sharpe_a, volatil, volatil_a, sortino_d, sortino_a)

console.print()
console.rule("[bold cyan]FIN DEL ANALISIS[/]", style="cyan")
console.print()


respuesta = console.input("[bold yellow]¿Correr analisis de estrategias (version final)? [y/n]: [/]").strip().lower()

if respuesta in ("y", "yes", "s", "si"):
    _correr_estrategias(directorio)
else:
    console.print("[dim]Saltando version final.[/]\n")
