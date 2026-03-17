import pathlib
from app.modelo_fundamental import graficas, Empresa , empresas_lit, FetchNews
from app.modelo_matematico import PromediosMoviles, VarianzaMovil, ModeloAR, TeoriaJuegos
from app.modelo_matematico import ratio_sharpe, volatilidad, sortino
from colorstreak import Logger
import polars as pl
import numpy as np
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text

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



directorio = EmpresasDirectory(empresas_tickers, graficas=True)

console.print()
console.rule("[bold cyan]ECONOMISTA INTELIGENTE[/]", style="cyan")
console.print(f"  Total de noticias clasificadas: [bold]{directorio.total_noticias:,}[/]")
console.print(f"  Empresas analizadas: [bold]{len(empresas_tickers)}[/]")
console.rule(style="cyan")
console.print()


def _color_senal(senal: str) -> str:
    if senal == "COMPRA":
        return "[bold green]COMPRA[/]"
    elif senal == "VENTA":
        return "[bold red]VENTA[/]"
    return "[dim]HOLD[/]"


def _color_tendencia(tendencia: str) -> str:
    if tendencia == "ALCISTA":
        return "[bold green]ALCISTA[/]"
    return "[bold red]BAJISTA[/]"


def _imprimir_empresa(ticker: str, pm, vm, ar, juego, num_noticias: int,
                      sharpe_d, sharpe_a, volatil, volatil_a, sortino_d, sortino_a) -> None:
    # ── Promedios Móviles ──
    pm_r = pm.resumen
    if pm_r.get("datos_insuficientes"):
        panel_pm = Panel("[dim]Datos insuficientes[/]", title="Promedios Moviles", border_style="yellow")
    else:
        ultima = pm_r.get("ultima_senal")
        ultima_txt = _color_senal(ultima["senal"]) if ultima else "[dim]ninguna[/]"
        sobre_corta = "[green]Si[/]" if pm_r.get("precio_sobre_sma_corta") else "[red]No[/]"
        sobre_larga = "[green]Si[/]" if pm_r.get("precio_sobre_sma_larga") else "[red]No[/]"

        tabla_pm = Table(show_header=False, box=None, padding=(0, 1))
        tabla_pm.add_column(style="dim")
        tabla_pm.add_column()
        tabla_pm.add_row("Precio actual", f"${pm_r['precio_actual']:,.2f}")
        tabla_pm.add_row("SMA 20", f"${pm_r.get('sma_20', 0):,.2f}")
        tabla_pm.add_row("SMA 50", f"${pm_r.get('sma_50', 0):,.2f}")
        tabla_pm.add_row("Precio > SMA20", sobre_corta)
        tabla_pm.add_row("Precio > SMA50", sobre_larga)
        tabla_pm.add_row("Senales totales", str(pm_r["total_senales"]))
        tabla_pm.add_row("Ultima senal", ultima_txt)
        panel_pm = Panel(tabla_pm, title="Promedios Moviles", border_style="blue")

    # ── Volatilidad & Bollinger ──
    vm_r = vm.resumen
    if vm_r.get("datos_insuficientes"):
        panel_vm = Panel("[dim]Datos insuficientes[/]", title="Volatilidad", border_style="yellow")
    else:
        estado = "[bold red]SOBRECOMPRADO[/]" if vm_r["sobrecomprado"] else \
                 "[bold green]SOBREVENDIDO[/]" if vm_r["sobrevendido"] else \
                 "[dim]Normal[/]"
        pos_b = vm_r["posicion_bollinger"]
        barra_len = 20
        pos_int = max(0, min(barra_len, int(pos_b * barra_len)))
        barra = "[green]" + "█" * pos_int + "[/][dim]" + "░" * (barra_len - pos_int) + "[/]"

        tabla_vm = Table(show_header=False, box=None, padding=(0, 1))
        tabla_vm.add_column(style="dim")
        tabla_vm.add_column()
        tabla_vm.add_row("Banda superior", f"${vm_r['banda_superior']:,.2f}")
        tabla_vm.add_row("Precio", f"${vm_r['precio']:,.2f}")
        tabla_vm.add_row("Banda inferior", f"${vm_r['banda_inferior']:,.2f}")
        tabla_vm.add_row("Bollinger", f"{barra} {pos_b:.1%}")
        tabla_vm.add_row("Volatilidad", f"{vm_r['volatilidad_actual']:.4f}")
        tabla_vm.add_row("Estado", estado)
        panel_vm = Panel(tabla_vm, title="Volatilidad & Bollinger", border_style="magenta")

    # ── Ratios Financieros ──
    tabla_rf = Table(show_header=False, box=None, padding=(0, 1))
    tabla_rf.add_column(style="dim")
    tabla_rf.add_column()
    tabla_rf.add_row("Sharpe diario", f"{sharpe_d:.4f}")
    tabla_rf.add_row("Sharpe anual", f"{sharpe_a:.4f}")
    tabla_rf.add_row("Volatilidad diaria", f"{volatil:.4f}")
    tabla_rf.add_row("Volatilidad anual", f"{volatil_a:.4f}")
    tabla_rf.add_row("Sortino diario", f"{sortino_d:.4f}")
    tabla_rf.add_row("Sortino anual", f"{sortino_a:.4f}")
    panel_rf = Panel(tabla_rf, title="Ratios Financieros", border_style="yellow")

    # ── Modelo AR ──
    ar_r = ar.resumen
    if not ar_r.get("ajustado"):
        panel_ar = Panel("[dim]No se pudo ajustar[/]", title="Modelo AR", border_style="yellow")
    else:
        preds = ar_r["prediccion_5_dias"]
        preds_txt = "  ".join(
            f"[green]+{p:.4f}[/]" if p > 0 else f"[red]{p:.4f}[/]"
            for p in preds
        )
        tabla_ar = Table(show_header=False, box=None, padding=(0, 1))
        tabla_ar.add_column(style="dim")
        tabla_ar.add_column()
        tabla_ar.add_row("Orden", f"AR({ar_r['orden']})")
        tabla_ar.add_row("Tendencia", _color_tendencia(ar_r["tendencia"]))
        tabla_ar.add_row("Intercepto", f"{ar_r['intercepto']:.6f}")
        tabla_ar.add_row("Pred. 5 dias", preds_txt)
        panel_ar = Panel(tabla_ar, title="Modelo AR", border_style="cyan")

    # ── Teoria de Juegos ──
    j_r = juego.resumen
    if not j_r.get("construido"):
        panel_tj = Panel("[dim]Sin datos para construir la matriz[/]", title="Teoria de Juegos", border_style="yellow")
    else:
        equilibrios = j_r["equilibrio_nash"]["equilibrios"]

        # Tabla de la matriz de pagos
        matriz_tabla = Table(title="Matriz de Pagos", show_lines=True, border_style="dim")
        matriz_tabla.add_column("Sent \\ Tec", style="bold", width=8)
        for est in ["COMPRA", "HOLD", "VENTA"]:
            matriz_tabla.add_column(est, justify="center", width=10)

        pagos = j_r["matriz_pagos"]
        for sent in ["COMPRA", "HOLD", "VENTA"]:
            fila = []
            for tec in ["COMPRA", "HOLD", "VENTA"]:
                val = pagos.get(f"({sent},{tec})", 0.0)
                if val > 0.01:
                    fila.append(f"[green]{val:+.4f}[/]")
                elif val < -0.01:
                    fila.append(f"[red]{val:+.4f}[/]")
                else:
                    fila.append(f"[dim]{val:+.4f}[/]")
            matriz_tabla.add_row(sent, *fila)

        # Equilibrios
        eq_lines = []
        for eq in equilibrios:
            eq_lines.append(
                f"  {_color_senal(eq['sentimiento'])} + {_color_senal(eq['tecnico'])} "
                f"= pago [bold]{eq['pago']:+.4f}[/]"
            )
        eq_txt = "\n".join(eq_lines) if eq_lines else "[dim]Ninguno[/]"

        panel_tj = Panel(
            Group(matriz_tabla, Text(""), Text.from_markup(f"[bold]Equilibrio Nash:[/]\n{eq_txt}")),
            title="Teoria de Juegos (Nash)",
            border_style="green",
        )

    # ── Imprimir todo ──
    console.print()
    console.rule(f"[bold yellow]{ticker}[/]  —  {num_noticias:,} noticias", style="yellow")
    console.print(Columns([panel_pm, panel_vm, panel_rf], equal=True, expand=True))
    console.print(Columns([panel_ar, panel_tj], equal=True, expand=True))


for ticker, empresa in directorio.empresas:
    informacion_empresa = empresa.all_info
    cierres  = empresa.cierres
    rendimiento_simple = empresa.rendimiento_simple
    rendimiento_logaritmico = empresa.rendimiento_log

    # Ratios financieros (del remoto)
    sharpe_d, sharpe_a = ratio_sharpe(rendimiento_simple)
    volatil, volatil_a = volatilidad(rendimiento_simple)
    sortino_d, sortino_a = sortino(rendimiento_simple)

    # 1. Promedios móviles
    pm = PromediosMoviles(cierres)

    # 2. Varianza y Bollinger
    vm = VarianzaMovil(cierres)

    # 3. Modelo AR sobre rendimientos logarítmicos
    ar = ModeloAR(rendimiento_logaritmico, orden=5)

    # 4. Teoría de juegos: sentimiento vs técnico
    sentimientos = informacion_empresa.select("sentiment").to_series().to_list()
    senales_sentimiento = [
        "COMPRA" if s == "positive" else "VENTA" if s == "negative" else "HOLD"
        for s in sentimientos
    ]
    senales_tecnicas_df = pm.tabla.select("senal").to_series().to_list()
    n = min(len(rendimiento_simple), len(senales_tecnicas_df), len(senales_sentimiento))
    juego = TeoriaJuegos(
        rendimientos=rendimiento_simple[:n],
        senales_tecnicas=senales_tecnicas_df[:n],
        senales_sentimiento=senales_sentimiento[:n],
    )

    _imprimir_empresa(ticker, pm, vm, ar, juego, informacion_empresa.height,
                      sharpe_d, sharpe_a, volatil, volatil_a, sortino_d, sortino_a)

console.print()
console.rule("[bold cyan]FIN DEL ANALISIS[/]", style="cyan")
console.print()
