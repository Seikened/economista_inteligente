# Economista Inteligente — Explicación del Proyecto

## ¿Qué es?

Un pipeline de análisis financiero que combina **inteligencia artificial** (NLP) con **modelos matemáticos** para generar señales de inversión sobre 19 empresas tecnológicas.

La idea central: no basta con saber "qué dice la prensa" (sentimiento) ni "qué dicen los números" (técnico). El proyecto **cruza ambas señales** usando teoría de juegos para encontrar la estrategia óptima.

---

## Arquitectura General

```
                         DATOS EXTERNOS
                    ┌──────────┬──────────┐
                    │ Finnhub  │ yfinance │
                    │ (noticias)│ (precios)│
                    └────┬─────┴────┬─────┘
                         │          │
                         ▼          ▼
              ┌─────────────────────────────┐
              │      MODELO FUNDAMENTAL     │
              │   app/modelo_fundamental/   │
              │                             │
              │  get_noticias.py            │
              │    └─ FetchNews             │
              │       Descarga noticias de  │
              │       Finnhub + precios de  │
              │       yfinance              │
              │       → noticias.json       │
              │                             │
              │  cliente_clasificador.py    │
              │    └─ Empresa               │
              │       Carga JSON → Polars   │
              │       Filtra por ticker     │
              │       Calcula rendimientos  │
              │       Aplica FinBERT        │
              │       Cache en Parquet      │
              │                             │
              │  modelo_clasificador.py     │
              │    └─ ClasificadorSentimientos│
              │       FinBERT (ProsusAI)    │
              │       headline + summary    │
              │       → sentimiento + prob  │
              └──────────────┬──────────────┘
                             │
                   ┌─────────┴─────────┐
                   │ precios  sentimiento│
                   │ cierres  pos/neu/neg│
                   └─────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │      MODELO MATEMATICO      │
              │   app/modelo_matematico/    │
              │                             │
              │  promedios_moviles.py       │
              │    └─ SMA, EMA, cruces      │
              │                             │
              │  varianza_movil.py          │
              │    └─ Bollinger, volatilidad │
              │                             │
              │  modelo_ar.py               │
              │    └─ AR(p), predicción      │
              │                             │
              │  teoria_juegos.py           │
              │    └─ Nash: sent vs técnico  │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │         DIRECTOR            │
              │        director.py          │
              │                             │
              │  EmpresasDirectory          │
              │    └─ Orquesta todo          │
              │                             │
              │  Visualización con Rich     │
              │    └─ Paneles por empresa    │
              │    └─ Tablas de Nash         │
              │    └─ Barras de Bollinger    │
              └─────────────────────────────┘
```

---

## Modelo Fundamental (NLP + Datos)

### ¿Qué hace?

1. **Descarga noticias** de los últimos 7 años desde Finnhub API para 19 tickers
2. **Enriquece** cada noticia con precio de cierre del día (yfinance)
3. **Clasifica sentimiento** de cada noticia usando FinBERT (modelo de HuggingFace pre-entrenado en texto financiero)
4. **Cachea** los resultados en archivos Parquet para no repetir clasificaciones

### FinBERT

Es un modelo BERT fine-tuned específicamente para texto financiero. Clasifica en 3 categorías:
- **positive** → la noticia es favorable para la acción
- **neutral** → no tiene impacto claro
- **negative** → la noticia es desfavorable

Se clasifica tanto el titular como el resumen de cada noticia y se promedian con pesos.

### Flujo de datos

```
Finnhub API → JSON crudo → Polars LazyFrame → filtro por ticker
    → FinBERT clasifica → merge con precios → Parquet (cache)
```

---

## Modelo Matemático (4 módulos)

### 1. Promedios Móviles (`promedios_moviles.py`)

**Clase:** `PromediosMoviles(cierres, ventana_corta=20, ventana_larga=50)`

Calcula:
- **SMA** (Simple Moving Average): promedio simple de los últimos N días
- **EMA** (Exponential Moving Average): promedio ponderado que da más peso a datos recientes
- **Señales de cruce**: cuando SMA corta cruza SMA larga

Señales:
- **COMPRA** (Golden Cross): SMA corta cruza por arriba de SMA larga → tendencia alcista
- **VENTA** (Death Cross): SMA corta cruza por abajo → tendencia bajista
- **HOLD**: sin cambio

```
SMA(20) = promedio(últimos 20 cierres)
EMA(20) = cierre_hoy × (2/21) + EMA_ayer × (1 - 2/21)
Señal   = cambio en signo(SMA_corta - SMA_larga)
```

---

### 2. Varianza Móvil (`varianza_movil.py`)

**Clase:** `VarianzaMovil(cierres, ventana=20, num_desviaciones=2.0)`

Calcula:
- **Bandas de Bollinger**: SMA ± 2 desviaciones estándar
- **Posición Bollinger**: dónde está el precio dentro de las bandas (0% a 100%)
- **Volatilidad**: desviación estándar rolling de los rendimientos

Diagnóstico:
- `posición > 100%` → **sobrecomprado** (precio inusualmente alto)
- `posición < 0%` → **sobrevendido** (precio inusualmente bajo)
- Alta volatilidad → señales menos confiables

```
Banda superior = SMA(20) + 2σ
Banda inferior = SMA(20) - 2σ
Posición = (precio - banda_inferior) / (banda_superior - banda_inferior)
```

---

### 3. Modelo AR (`modelo_ar.py`)

**Clase:** `ModeloAR(rendimientos, orden=5)`

Un modelo **autorregresivo** que predice el rendimiento futuro basándose en rendimientos pasados.

```
y(t) = c + φ₁·y(t-1) + φ₂·y(t-2) + ... + φ₅·y(t-5)
```

- Se ajusta con **mínimos cuadrados ordinarios** (OLS) usando solo numpy
- Usa **rendimientos logarítmicos** porque son aditivos y estadísticamente estables
- Predice los próximos **5 días** de rendimiento
- Clasifica tendencia: **ALCISTA** si la predicción promedio > 0, **BAJISTA** si < 0

No requiere statsmodels ni ninguna dependencia extra.

---

### 4. Teoría de Juegos (`teoria_juegos.py`)

**Clase:** `TeoriaJuegos(rendimientos, senales_tecnicas, senales_sentimiento)`

Este es el módulo que **une todo**. Modela la decisión de inversión como un juego de 2 jugadores:

| | COMPRA (tec) | HOLD (tec) | VENTA (tec) |
|---|---|---|---|
| **COMPRA (sent)** | pago | pago | pago |
| **HOLD (sent)** | pago | pago | pago |
| **VENTA (sent)** | pago | pago | pago |

**Jugadores:**
- **Jugador 1 (filas)**: Señal Fundamental → sentimiento FinBERT mapeado a COMPRA/HOLD/VENTA
- **Jugador 2 (columnas)**: Señal Técnica → señales de cruce de promedios móviles

**Cómo se construye la matriz de pagos:**
1. Para cada día se tiene: rendimiento real, señal técnica y señal de sentimiento
2. Se agrupan por combinación (ej: sentimiento=COMPRA, técnico=HOLD)
3. El pago = **rendimiento real promedio** cuando esa combinación ocurrió

**Equilibrio de Nash:**
- Se busca la celda donde ambos jugadores están en su **mejor respuesta mutua**
- Ninguno ganaría cambiando su estrategia unilateralmente
- Si no hay equilibrio puro, se reporta la combinación con mayor pago global

**¿Por qué es útil?**
En vez de elegir ciegamente una señal, Nash dice: "históricamente, cuando el sentimiento decía COMPRA y el técnico decía HOLD, el rendimiento promedio fue X". Te da la **combinación óptima**.

---

## Integración en `director.py`

El archivo `director.py` es el orquestador central. Tiene dos partes:

### Parte 1: Carga de datos (`EmpresasDirectory`)

```python
directorio = EmpresasDirectory(empresas_tickers, graficas=True)
```

Esto ejecuta:
1. Verifica si existe `noticias.json`, si no lo crea descargando de Finnhub
2. Para cada ticker: carga noticias, clasifica con FinBERT, cachea en Parquet
3. Si `graficas=True`, genera gráficas de distribución de sentimiento

### Parte 2: Análisis matemático (loop por empresa)

Para cada empresa se ejecutan los 4 módulos en secuencia:

```python
for ticker, empresa in directorio.empresas:
    cierres = empresa.cierres
    rendimiento_simple = empresa.rendimiento_simple
    rendimiento_logaritmico = empresa.rendimiento_log

    pm    = PromediosMoviles(cierres)                    # SMA, EMA, cruces
    vm    = VarianzaMovil(cierres)                       # Bollinger, volatilidad
    ar    = ModeloAR(rendimiento_logaritmico, orden=5)   # predicción AR(5)

    # Mapear sentimientos a señales
    sentimientos → COMPRA / HOLD / VENTA

    # Cruzar con señales técnicas en Nash
    juego = TeoriaJuegos(rendimientos, señales_técnicas, señales_sentimiento)
```

### Parte 3: Visualización con Rich

Cada empresa se muestra con 4 paneles en terminal:

```
──────────────── NVDA — 3,180 noticias ────────────────
╭─ Promedios Moviles ─╮  ╭─ Volatilidad & Bollinger ─╮
│ Precio: $181.81      │  │ Bollinger: ██████░░░ 31.4% │
│ SMA 20: $184.78      │  │ Volatilidad: 0.0197        │
│ Señal: COMPRA        │  │ Estado: Normal              │
╰──────────────────────╯  ╰────────────────────────────╯
╭──── Modelo AR ───────╮  ╭── Teoria de Juegos (Nash) ─╮
│ Tendencia: ALCISTA   │  │ Matriz de pagos 3x3        │
│ Pred: +0.01 -0.00 ..│  │ Equilibrio: COMPRA + HOLD  │
╰──────────────────────╯  ╰────────────────────────────╯
```

Los colores indican:
- **Verde**: valores positivos, COMPRA, ALCISTA, precio sobre SMA
- **Rojo**: valores negativos, VENTA, BAJISTA, sobrecomprado
- **Dim/gris**: HOLD, valores neutros, datos insuficientes

---

## Dependencias clave

| Librería | Para qué |
|---|---|
| `polars` | DataFrames con evaluación lazy (rápido) |
| `numpy` | Álgebra lineal (OLS del AR, matrices de Nash) |
| `transformers` + `torch` | FinBERT para clasificar sentimiento |
| `yfinance` | Precios de cierre históricos |
| `matplotlib` | Gráficas de distribución de sentimiento |
| `rich` | Visualización bonita en terminal |

---

## Estructura de archivos

```
economista_inteligente/
├── director.py                           # Orquestador principal
├── app/
│   ├── modelo_fundamental/               # NLP + datos
│   │   ├── get_noticias.py               #   FetchNews (Finnhub + yfinance)
│   │   ├── cliente_clasificador.py       #   Empresa (Polars + cache Parquet)
│   │   └── modelo_clasificador.py        #   ClasificadorSentimientos (FinBERT)
│   └── modelo_matematico/                # Modelos matemáticos
│       ├── promedios_moviles.py          #   SMA, EMA, Golden/Death Cross
│       ├── varianza_movil.py             #   Bollinger Bands, volatilidad
│       ├── modelo_ar.py                  #   AR(p) con OLS
│       └── teoria_juegos.py              #   Matriz de pagos, Nash
├── data/
│   ├── parquets_finanzas/                # Cache de datos procesados
│   └── graficas/                         # Gráficas generadas
├── noticias.json                         # Cache de noticias (runtime)
└── pyproject.toml                        # Dependencias (UV)
```

---

## Flujo completo resumido

```
1. Descarga noticias (Finnhub) + precios (yfinance)
2. Clasifica sentimiento (FinBERT): positive / neutral / negative
3. Calcula promedios móviles (SMA/EMA) → señales COMPRA/HOLD/VENTA
4. Calcula volatilidad (Bollinger) → confianza en las señales
5. Predice rendimientos (AR) → tendencia ALCISTA/BAJISTA
6. Cruza sentimiento vs técnico (Nash) → estrategia óptima
7. Muestra todo bonito en terminal (Rich)
```
