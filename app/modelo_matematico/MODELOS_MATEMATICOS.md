# Modelos Matemáticos — Explicación

## Visión general

Cuatro módulos que transforman datos crudos (precios + sentimiento) en señales de decisión financiera. Cada uno alimenta al siguiente:

```
Precios de cierre
    │
    ├──→ 1. Promedios Móviles (SMA, EMA, señales de cruce)
    │
    ├──→ 2. Varianza Móvil (Bollinger Bands, volatilidad)
    │
    ├──→ 3. Modelo AR (predicción de rendimientos)
    │
    └──→ 4. Teoría de Juegos (Nash: sentimiento vs técnico)
              ↑                        ↑
         señales técnicas      señales de sentimiento
         (de módulo 1)         (de FinBERT)
```

---

## 1. Promedios Móviles (`promedios_moviles.py`)

### ¿Qué hace?
Calcula SMA y EMA con ventanas corta (20 días) y larga (50 días), y detecta señales de cruce.

### ¿Por qué?
El cruce de promedios móviles es uno de los indicadores técnicos más usados. Cuando la media corta cruza por arriba de la larga (Golden Cross) indica tendencia alcista. Cuando cruza por abajo (Death Cross) indica tendencia bajista.

### Fórmulas
- **SMA(n)** = (1/n) * Σ precios últimos n días
- **EMA(n)** = precio_hoy * (2/(n+1)) + EMA_ayer * (1 - 2/(n+1))
- **Señal** = signo(SMA_corta - SMA_larga) comparado con el día anterior

---

## 2. Varianza Móvil (`varianza_movil.py`)

### ¿Qué hace?
Calcula Bandas de Bollinger y volatilidad rolling de los rendimientos.

### ¿Por qué?
Los promedios móviles solos no dicen qué tan "confiable" es la señal. La varianza agrega contexto:
- **Alta volatilidad** = señales menos confiables, mercado nervioso
- **Precio fuera de las bandas** = movimiento extremo (sobrecompra/sobreventa)

### Fórmulas
- **Banda superior** = SMA(20) + 2σ
- **Banda inferior** = SMA(20) - 2σ
- **Posición Bollinger** = (precio - banda_inferior) / (banda_superior - banda_inferior)
  - `> 1.0` → sobrecomprado
  - `< 0.0` → sobrevendido
- **Volatilidad** = desviación estándar rolling de los rendimientos diarios

---

## 3. Modelo AR (`modelo_ar.py`)

### ¿Qué hace?
Ajusta un modelo autorregresivo AR(p) sobre rendimientos logarítmicos y predice los próximos 5 días.

### ¿Por qué?
Un AR modela la dependencia temporal: el rendimiento de hoy depende de los rendimientos pasados. Es el modelo base de series de tiempo financieras. Usamos rendimientos log porque son aditivos y más estables estadísticamente.

### Fórmulas
- **AR(p)**: y(t) = c + φ₁·y(t-1) + φ₂·y(t-2) + ... + φₚ·y(t-p) + ε
- **Ajuste**: mínimos cuadrados ordinarios (OLS)
- **Predicción**: se aplica la ecuación recursivamente con los últimos p valores

### Implementación
Se resuelve sin dependencias externas usando `numpy.linalg.lstsq` directamente. No necesita statsmodels.

---

## 4. Teoría de Juegos (`teoria_juegos.py`)

### ¿Qué hace?
Modela la decisión de inversión como un juego de 2 jugadores:
- **Jugador 1 (filas)**: Señal Fundamental (sentimiento FinBERT) → COMPRA / HOLD / VENTA
- **Jugador 2 (columnas)**: Señal Técnica (cruce de promedios) → COMPRA / HOLD / VENTA

### ¿Por qué?
En vez de seguir ciegamente una sola señal, Nash nos dice cuándo ambas señales coinciden en la mejor estrategia. El equilibrio es el punto donde ningún jugador gana cambiando su estrategia unilateralmente.

### Cómo se construye la matriz de pagos
1. Para cada día tenemos: rendimiento real, señal técnica y señal de sentimiento
2. Se agrupan por cada combinación (ej: sentimiento=COMPRA, técnico=HOLD)
3. El pago de esa celda = rendimiento promedio real cuando esa combinación ocurrió
4. Celdas sin datos = 0

### Equilibrio de Nash (estrategias puras)
1. Para cada columna, encontrar la mejor fila (mejor respuesta del sentimiento)
2. Para cada fila, encontrar la mejor columna (mejor respuesta del técnico)
3. Si una celda es mejor respuesta para ambos → equilibrio de Nash
4. Si no hay equilibrio puro, se reporta la combinación con mayor pago global

### Cómo se mapean las señales
- **Sentimiento**: positive → COMPRA, negative → VENTA, neutral → HOLD
- **Técnico**: se toma directamente del módulo de promedios móviles (COMPRA/HOLD/VENTA)

---

## Integración en `director.py`

Para cada empresa, el loop ejecuta los 4 módulos secuencialmente:

```python
pm    = PromediosMoviles(cierres)           # señales de cruce
vm    = VarianzaMovil(cierres)              # volatilidad y Bollinger
ar    = ModeloAR(rendimiento_log, orden=5)  # predicción AR(5)
juego = TeoriaJuegos(rendimientos, señales_técnicas, señales_sentimiento)  # Nash
```

Cada módulo expone `.resumen` (dict) para logging rápido y `.tabla` (DataFrame) para análisis detallado.
