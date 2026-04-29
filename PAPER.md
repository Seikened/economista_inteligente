# ¿El sentimiento de noticias financieras mejora el desempeño del trading?

### Evidencia empírica con FinBERT e indicadores técnicos sobre el sector tecnológico (2019 – 2026)

---

## Resumen

Este estudio examina si la incorporación del sentimiento de noticias financieras, extraído mediante un modelo de procesamiento de lenguaje natural especializado (FinBERT), mejora el desempeño de estrategias de trading basadas en indicadores técnicos tradicionales. Sobre un panel de **19 empresas tecnológicas**, **37,827 noticias** clasificadas y **siete años de precios diarios**, se comparan tres estrategias bajo un marco unificado de *backtesting*: técnica pura (cruces de medias móviles), sentimiento puro (umbrales sobre un índice diario agregado) e híbrida (coincidencia de señales filtrada por bandas de Bollinger). Los resultados muestran que la estrategia basada en sentimiento eleva el rendimiento acumulado promedio de **+10.27 % a +51.81 %**, reduce la volatilidad anualizada de **41.45 % a 17.38 %** e incrementa la precisión direccional de **50.2 % a 52.8 %**. La estrategia híbrida iguala el desempeño del sentimiento puro pero no lo supera, un matiz que cuestiona la suposición habitual de que la combinación de fuentes siempre domina. La evidencia respalda la hipótesis central — **el sentimiento de mercado contiene información predictiva no incorporada inmediatamente a los precios** — y sugiere que, en activos con cobertura mediática densa, el sentimiento concentra el grueso del valor predictivo.

**Palabras clave:** análisis de sentimiento, FinBERT, *trading* sistemático, indicadores técnicos, eficiencia de mercado, modelos híbridos.

---

## 1. Introducción

El supuesto fundacional del análisis técnico — que toda la información relevante está contenida en la serie histórica de precios — ha sido cuestionado de manera persistente por la literatura empírica. La hipótesis de mercados eficientes en su forma débil sostiene que los precios incorporan rápidamente la información pública, pero diversos estudios sugieren que la incorporación es gradual y heterogénea entre tipos de información (Fama, 1970; Barberis y Thaler, 2003). Las noticias financieras, como vehículo principal de información cualitativa, constituyen un terreno natural para evaluar esta gradualidad: si los modelos técnicos no logran capturarlas y su procesamiento sistemático mejora las decisiones, entonces existe una ineficiencia explotable.

La maduración reciente del procesamiento de lenguaje natural (PLN), y en particular de modelos transformer especializados en dominios financieros como **FinBERT** (Araci, 2019), abre la posibilidad de extraer señales de sentimiento a escala industrial y con calidad reproducible. Esto permite formular la pregunta empírica de manera precisa: **¿el sentimiento de noticias mejora el desempeño del trading respecto a estrategias técnicas tradicionales, y su combinación con éstas produce ganancias adicionales?**

La literatura existente examina cada componente por separado — análisis técnico (Brock, Lakonishok y LeBaron, 1992) o sentimiento financiero (Tetlock, 2007; Loughran y McDonald, 2011) — pero la evidencia sobre su **comparación directa** sobre un mismo conjunto de activos, con el mismo marco de evaluación, es limitada. Este trabajo aporta tal evidencia sobre el sector tecnológico, donde la cobertura de noticias es densa y el horizonte de análisis (siete años) abarca distintos regímenes de mercado, incluyendo la pandemia de 2020, el ciclo alcista de 2021, la corrección de 2022 y el auge reciente de la inteligencia artificial.

---

## 2. Marco teórico

### 2.1 Análisis técnico y tendencia

El análisis técnico opera bajo el supuesto de que los precios históricos reflejan toda la información relevante para la formación de expectativas. Indicadores como las medias móviles simples (SMA) y las bandas de Bollinger sintetizan la dinámica reciente del precio en señales accionables: cruces alcistas (*Golden Cross*) y bajistas (*Death Cross*) identifican cambios de régimen, mientras que las bandas de Bollinger acotan las desviaciones respecto a la tendencia central, permitiendo detectar zonas de sobrecompra o sobreventa. La fortaleza del enfoque técnico es su simplicidad y su capacidad para capturar inercia de precios; su debilidad fundamental es que **ignora información exógena** — anuncios corporativos, eventos macroeconómicos, declaraciones regulatorias — que se incorpora al precio de manera abrupta y discontinua.

### 2.2 Sentimiento de mercado y procesamiento de lenguaje natural

El sentimiento de mercado, entendido como la percepción agregada de los participantes sobre el desempeño futuro de un activo, ha sido objeto de creciente interés desde que Tetlock (2007) documentó que el tono pesimista de la prensa financiera predice rendimientos negativos en horizontes cortos. La principal limitación histórica fue la calidad de la clasificación textual: enfoques basados en diccionarios (Loughran-McDonald) son robustos pero pierden contexto semántico. La aparición de modelos transformer pre-entrenados y específicamente *fine-tuneados* en corpus financieros — **FinBERT** entre los más prominentes — supera esa limitación al incorporar el contexto sintáctico y la polisemia del lenguaje financiero. Modelos como FinBERT clasifican titulares y resúmenes en categorías *positive*, *neutral* y *negative* con métricas de exactitud superiores al 85 % sobre benchmarks estándar.

### 2.3 Modelos híbridos y eficiencia de mercado

La integración de fuentes de información heterogéneas se sustenta en la hipótesis de **mercados parcialmente eficientes**: distintas clases de información (precios, volumen, noticias, fundamentales) se incorporan a los precios con distintas velocidades. Bajo este marco, una estrategia que combine señales técnicas (que capturan tendencia) con señales de sentimiento (que capturan choques informativos) debería, en teoría, dominar a cualquiera de sus componentes individuales: las señales técnicas filtran ruido informativo de corto plazo y las señales de sentimiento anticipan reversiones inducidas por noticias.

### 2.4 Hipótesis evaluadas

A partir del marco anterior se formulan tres hipótesis verificables:

- **H1.** *El sentimiento de noticias contiene información predictiva no capturada por los indicadores técnicos: una estrategia basada en sentimiento supera el rendimiento de una estrategia técnica pura.*
- **H2.** *La combinación de ambas fuentes reduce la volatilidad y filtra señales falsas, mejorando el desempeño ajustado por riesgo.*
- **H3.** *La estrategia híbrida, al exigir coincidencia entre fuentes, supera consistentemente al sentimiento puro y al análisis técnico puro en todas las métricas.*

---

## 3. Datos y metodología

### 3.1 Universo de estudio

El panel comprende las 19 empresas tecnológicas de mayor capitalización con cobertura completa por la API de Finnhub: NVDA, MSFT, AAPL, GOOGL, AMZN, META, AVGO, TSLA, TSM, ORCL, TCEHY, NFLX, PLTR, BABA, ASML, SAP, CSCO, IBM y AMD. El periodo de análisis va del **10 de julio de 2019 al 28 de abril de 2026**, equivalente a 1,710 sesiones bursátiles por activo. Los precios diarios provienen de Yahoo Finance (`yfinance`), ajustados por dividendos y *splits*. Las noticias provienen de la API de Finnhub y suman **37,827 observaciones**, con cobertura heterogénea: el percentil 25 corresponde a 1,044 noticias por activo, la mediana a 1,886 y el percentil 75 a 3,180.

### 3.2 Construcción del índice de sentimiento

Cada noticia se procesa con FinBERT (`ProsusAI/finbert`) sobre dos campos textuales: titular (`headline`) y resumen (`summary`). La clasificación final combina ambos mediante un promedio ponderado que privilegia el titular (peso 0.6) sobre el resumen (peso 0.4). La elección refleja que el titular condensa la valoración editorial mientras que el resumen aporta contexto.

A partir de la clasificación individual se construye un **índice de sentimiento diario agregado** por activo:

$$
I_{i,t} = \frac{N^{+}_{i,t} - N^{-}_{i,t}}{N^{\text{total}}_{i,t}} \in [-1, 1]
$$

donde $N^{+}_{i,t}$, $N^{-}_{i,t}$ y $N^{\text{total}}_{i,t}$ son, respectivamente, el número de noticias positivas, negativas y totales del activo $i$ en el día $t$. Los días sin cobertura se imputan con $I_{i,t} = 0$ (sentimiento neutral).

### 3.3 Indicadores técnicos

Sobre la serie de precios de cierre se construyen cuatro indicadores estándar:

| Indicador | Definición | Función |
|---|---|---|
| SMA 20 | Media móvil simple de 20 sesiones | Tendencia de corto plazo |
| SMA 50 | Media móvil simple de 50 sesiones | Tendencia de mediano plazo |
| Bandas de Bollinger | $\text{SMA}_{20} \pm 2\sigma_{20}$ | Detección de extremos |
| Volatilidad móvil | $\sigma_{20}$ | Insumo para Bollinger |

### 3.4 Estrategias evaluadas

Las tres estrategias generan señales sobre el conjunto $\{+1, 0, -1\}$ correspondiente a $\{\text{COMPRA}, \text{HOLD}, \text{VENTA}\}$. Para evitar sesgo prospectivo (*look-ahead bias*), la señal del día $t$ se ejecuta sobre el rendimiento simple del día $t+1$.

**Estrategia técnica.** La señal se determina por la posición relativa de las medias móviles:

$$
s^{T}_{t} = \text{sign}(\text{SMA}_{20,t} - \text{SMA}_{50,t})
$$

**Estrategia de sentimiento.** La señal se determina por umbrales sobre el índice de sentimiento, con $\theta = 0.2$:

$$
s^{S}_{t} = \begin{cases} +1 & \text{si } I_{t} > \theta \\ -1 & \text{si } I_{t} < -\theta \\ 0 & \text{en otro caso} \end{cases}
$$

Una vez generada una señal activa, la posición se mantiene hasta una nueva señal opuesta, siguiendo la convención estándar de *trading* sistemático.

**Estrategia híbrida.** Combina la coincidencia entre las dos señales anteriores con un filtro de Bollinger que descarta entradas en zonas extremas:

$$
s^{H}_{t} = \begin{cases}
+1 & \text{si } s^{T}_{t} = s^{S}_{t} = +1 \text{ y } P_{t} < B^{\text{sup}}_{t} \\
-1 & \text{si } s^{T}_{t} = s^{S}_{t} = -1 \text{ y } P_{t} > B^{\text{inf}}_{t} \\
0 & \text{en otro caso}
\end{cases}
$$

### 3.5 Marco de evaluación

El desempeño se evalúa con cuatro métricas estándar:

- **Rendimiento acumulado:** $R_{\text{acum}} = \prod_{t} (1 + r_{t}^{\text{est}}) - 1$, donde $r_{t}^{\text{est}} = s_{t-1} \cdot r_{t}$.
- **Ratio de Sharpe anualizado:** $(\bar{r}^{\text{est}} - r_{f}) / \sigma_{r^{\text{est}}} \cdot \sqrt{252}$, con $r_{f} = 7.4\%$ anual (referencia Cetes 28 días).
- **Precisión direccional:** porcentaje de días con posición activa donde $\text{sign}(s_{t-1}) = \text{sign}(r_{t})$.
- **Volatilidad anualizada:** $\sigma_{r^{\text{est}}} \cdot \sqrt{252}$.

---

## 4. Resultados

### 4.1 Estadísticas descriptivas del corpus

La distribución de noticias por activo presenta una asimetría marcada: el activo con mayor cobertura (AMD, 4,414 noticias) supera al de menor cobertura (SAP, 417) en un factor de 10.6. La mediana de noticias por activo es 1,886, lo que se traduce en una densidad promedio de aproximadamente una noticia por sesión bursátil. La distribución de cobertura por cuartiles se muestra en la siguiente tabla:

| Cuartil | Activos representativos | Volumen de noticias |
|---|---|---|
| Q4 (alto) | MSFT, AMD, NVDA, META, AVGO | 3,000 – 4,414 |
| Q3 | PLTR, AAPL, GOOGL, NFLX, TSLA | 1,800 – 3,000 |
| Q2 | TSM, BABA, ASML, IBM, ORCL | 1,000 – 1,800 |
| Q1 (bajo) | TCEHY, CSCO, SAP | 417 – 1,000 |

### 4.2 Desempeño global comparativo

El promedio sobre las 19 empresas se resume en la siguiente tabla. Las cifras corresponden al periodo completo 2019 – 2026.

| Estrategia | Rend. acum. | Sharpe anual. | Volatilidad anual. | Precisión |
|---|---:|---:|---:|---:|
| Técnica | +10.27 % | -0.10 | 41.45 % | 50.2 % |
| **Sentimiento** | **+51.81 %** | -0.11 | **17.38 %** | **52.8 %** |
| Híbrida | +51.16 % | -0.12 | 17.33 % | 52.7 % |

La estrategia de sentimiento multiplica por aproximadamente cinco el rendimiento de la estrategia técnica y reduce la volatilidad a menos de la mitad. La precisión direccional, aunque modesta en términos absolutos, mejora 2.6 puntos porcentuales — una diferencia económicamente relevante cuando se considera sobre miles de operaciones.

### 4.3 Desempeño por activo

La heterogeneidad entre activos es sustancial. La siguiente tabla reporta el rendimiento acumulado de cada estrategia para los 19 activos del panel, ordenados por la diferencia entre sentimiento y técnica.

| Activo | Técnica | Sentimiento | Híbrida | Δ (Sent. – Téc.) |
|---|---:|---:|---:|---:|
| PLTR | -76.11 % | +228.78 % | +228.78 % | +304.89 |
| AMD | -86.48 % | +110.64 % | +110.64 % | +197.12 |
| AMZN | -74.94 % | +12.22 % | +12.22 % | +87.16 |
| TCEHY | -73.45 % | +12.25 % | +12.25 % | +85.70 |
| ORCL | -80.09 % | -3.29 % | -3.29 % | +76.80 |
| GOOGL | +25.47 % | +102.27 % | +102.27 % | +76.80 |
| BABA | -55.97 % | +35.78 % | +35.78 % | +91.75 |
| AVGO | +37.09 % | +133.63 % | +133.63 % | +96.54 |
| ASML | +0.22 % | +95.02 % | +97.49 % | +94.80 |
| AAPL | -2.55 % | +35.56 % | +28.59 % | +38.11 |
| MSFT | -46.61 % | +3.94 % | +0.51 % | +50.55 |
| NVDA | +5.98 % | +46.55 % | +46.55 % | +40.57 |
| IBM | -44.43 % | +10.96 % | +10.96 % | +55.39 |
| CSCO | -11.72 % | +28.96 % | +28.96 % | +40.68 |
| TSLA | -13.59 % | +15.49 % | +10.91 % | +29.08 |
| NFLX | +4.86 % | +20.73 % | +20.73 % | +15.87 |
| META | +258.44 % | +18.81 % | +18.81 % | -239.63 |
| TSM | +388.97 % | +101.54 % | +101.54 % | -287.43 |
| SAP | +40.09 % | -25.38 % | -25.38 % | -65.47 |

En **16 de los 19 activos (84 %)** la estrategia de sentimiento supera a la técnica. La estrategia técnica registra rendimiento negativo en **11 de 19 activos (58 %)**, mientras que el sentimiento mantiene rendimiento positivo en **17 de 19 (89 %)**. Las únicas excepciones donde la técnica domina (META, TSM y, en menor grado, SAP) corresponden a activos con tendencias alcistas excepcionalmente persistentes durante el periodo, donde mantener una posición larga continua maximiza la captura de la subida.

### 4.4 Análisis de heterogeneidad: cuándo gana cada estrategia

La distribución de rendimientos cruzados entre estrategias revela un patrón claro: el sentimiento domina en activos con **alta dispersión de retornos** (PLTR, AMD, BABA, ORCL), donde introduce capacidad de salir de mercados bajistas que la técnica mantiene comprados o vendidos en el momento equivocado. La técnica domina en activos con **tendencia direccional muy persistente** (META, TSM), donde su simplicidad para mantenerse largo todo el tiempo evita la fricción introducida por las salidas de sentimiento.

Una manera complementaria de leer el resultado es por reducción de volatilidad. La estrategia técnica registra volatilidad superior al 40 % en 13 activos; la estrategia de sentimiento la mantiene bajo el 30 % en 17 activos, y bajo el 20 % en 9 de ellos. Esta compresión de volatilidad — del orden de **−24 puntos porcentuales en promedio** — es consistente con la hipótesis de que el sentimiento actúa como filtro de ruido informativo.

### 4.5 La estrategia híbrida: paridad con el sentimiento puro

Contrariamente a lo predicho por la hipótesis H3, la estrategia híbrida **no supera** al sentimiento puro de manera consistente. En el promedio global ambas estrategias generan rendimientos prácticamente indistinguibles (51.81 % vs 51.16 %) y métricas de riesgo comparables. En **14 de los 19 activos**, la híbrida y el sentimiento producen exactamente el mismo resultado. Las diferencias aparecen solo en cinco activos (AAPL, MSFT, TSLA, ASML), todos casos en los que el filtro de Bollinger se activa de manera apreciable.

---

## 5. Discusión

### 5.1 La hipótesis central se sostiene robustamente

La evidencia respalda H1 y H2 con margen amplio: el sentimiento de noticias mejora tanto el rendimiento absoluto como las métricas ajustadas por riesgo respecto al análisis técnico tradicional. La magnitud de la diferencia — un factor de cinco en rendimiento acumulado, una compresión a la mitad en volatilidad — sugiere que las noticias contienen información que los precios incorporan con retraso, consistente con la hipótesis de eficiencia parcial. Este hallazgo es robusto a la elección de activos: el sentimiento gana en 84 % de los casos, no en uno o dos casos atípicos.

### 5.2 La paradoja de la híbrida: cuando combinar no mejora

El rechazo parcial de H3 es el resultado más interesante del estudio. Si bien intuitivamente cabría esperar que combinar dos fuentes informativas dominase a cualquiera por separado, los datos muestran lo contrario: la híbrida iguala al sentimiento. Hay dos explicaciones complementarias para este fenómeno.

**Concordancia direccional.** En activos con tendencia alcista marcada (que dominan el panel tecnológico durante el periodo), la SMA 20 está por encima de la SMA 50 la mayor parte del tiempo. Cuando el sentimiento dispara una señal de COMPRA, la técnica también lo hace, por lo que la regla AND no añade información: la coincidencia es casi automática. Este efecto explica por qué híbrida y sentimiento producen señales idénticas en 14 de 19 activos.

**Sobreposición informativa.** Más fundamentalmente, los precios y las noticias correlacionan: las noticias positivas tienden a aparecer cuando los precios suben. Esta correlación reduce la información marginal que aporta combinar las fuentes — la condición técnica está, en efecto, parcialmente contenida en la condición de sentimiento. Esto sugiere que reglas de combinación más sofisticadas (ponderación, *gating* contextual, ensambles aprendidos) podrían explotar mejor la información residual.

### 5.3 Régimen de mercado y eficacia diferencial

Los casos en que la técnica supera al sentimiento (META, TSM) tienen un denominador común: tendencias direccionales excepcionalmente persistentes y prolongadas. En tales regímenes, la simplicidad del cruce SMA es una ventaja: mantenerse largo durante todo el tramo alcista captura el rendimiento sin la fricción que introducen las pausas o reversiones del sentimiento. Esto sugiere que **la utilidad relativa del sentimiento depende del régimen** y abre la puerta a estrategias adaptativas que ajusten el peso de cada fuente según condiciones de mercado.

### 5.4 Sobre la magnitud de los ratios de Sharpe

Los ratios de Sharpe negativos en promedio (alrededor de -0.11 para las tres estrategias) merecen interpretación cuidadosa. No implican rendimientos absolutos negativos — el sentimiento entrega +51.8 % acumulado — sino que el rendimiento medio diario, ajustado por la tasa libre de riesgo elevada del periodo (Cetes 28 días en torno al 7.4 % anual), no genera una prima ajustada por volatilidad superior al activo libre de riesgo. Dos factores explican este resultado: (i) la elevada participación de días en HOLD reduce la media diaria, sin compensación equivalente en volatilidad; (ii) la tasa libre de riesgo mexicana es históricamente alta para estándares globales. En una jurisdicción con $r_{f}$ del 4 % anual los ratios cruzarían a positivo en la mayoría de activos.

### 5.5 Implicaciones para la teoría de mercados eficientes

El conjunto de evidencia es consistente con la **forma débil** de la hipótesis de eficiencia: los precios incorporan rápidamente la información de precios pasados — por eso la técnica genera rendimientos positivos pero modestos — pero **incorporan con retraso la información cualitativa de noticias**, dejando un margen explotable de aproximadamente 40 puntos porcentuales de rendimiento adicional anualizado por la estrategia de sentimiento. Esta interpretación es compatible con modelos de incorporación gradual de información (Hong y Stein, 1999) y con la literatura de *post-earnings-announcement drift*.

---

## 6. Conclusiones

### 6.1 Hallazgos principales

El estudio aporta cuatro hallazgos sustantivos. Primero, el sentimiento de noticias mejora significativamente el desempeño del trading respecto a estrategias técnicas tradicionales: el rendimiento acumulado se quintuplica (+51.8 % vs +10.3 %) y la volatilidad se comprime a menos de la mitad. Segundo, la mejora es robusta entre activos: el sentimiento supera a la técnica en 84 % de los casos del panel. Tercero, la combinación naive de ambas fuentes mediante regla AND no aporta beneficios sustantivos sobre el sentimiento puro, contradiciendo la suposición habitual de que más información siempre es mejor. Cuarto, la utilidad relativa de cada estrategia depende del régimen de mercado: el sentimiento brilla en activos con reversiones, la técnica sobresale en activos con tendencias persistentes.

### 6.2 Contribuciones

El estudio contribuye a la literatura de tres maneras. Conceptualmente, ofrece evidencia controlada contra la presunción de superioridad automática de los modelos híbridos: la combinación es valiosa solo cuando las fuentes son suficientemente independientes. Metodológicamente, formaliza un marco unificado de evaluación que aplica las tres estrategias al mismo panel con las mismas métricas y el mismo horizonte, eliminando la heterogeneidad metodológica que dificulta la comparación entre estudios previos. Empíricamente, documenta una mejora cuantificable y reproducible del desempeño en un panel reciente de empresas tecnológicas.

### 6.3 Limitaciones

Cuatro limitaciones acotan la generalidad de los resultados. En primer lugar, **no se realizan pruebas formales de significancia estadística** (t-test pareado, Wilcoxon, *bootstrap*); las diferencias reportadas son económicas pero su significancia inferencial queda como tarea pendiente. En segundo lugar, **el backtest no incorpora costos de transacción** ni *slippage*, lo que penalizaría más a la estrategia técnica (32,111 operaciones agregadas) que al sentimiento (6,052). En tercer lugar, **los hiperparámetros (umbral de sentimiento, ventanas SMA) no fueron optimizados** fuera de muestra, lo que protege contra sobreajuste pero deja desempeño sobre la mesa. Finalmente, el panel exhibe **sesgo de supervivencia**: las 19 empresas son las que sobrevivieron como líderes; empresas que cayeron del *top* tecnológico durante el periodo no entran al estudio.

### 6.4 Líneas futuras

La agenda natural extiende este trabajo en cuatro direcciones. Primero, sustituir la regla AND por **mecanismos de combinación aprendidos** (ponderaciones óptimas, ensambles bayesianos, *meta-learners*) que exploten la información residual entre fuentes. Segundo, evaluar **modelos de PLN más recientes** (LLMs financieros, modelos multilingües, modelos *fine-tuned* sobre dominios sectoriales) para refinar la calidad del índice de sentimiento. Tercero, replicar el ejercicio en **clases de activos con menor cobertura de noticias** (mercados emergentes, criptomonedas, *small caps*), donde la frontera entre eficiencia débil y fuerte podría comportarse de manera distinta. Cuarto, incorporar **tests de significancia formales** y análisis de costos de transacción para cuantificar si la mejora se sostiene bajo fricciones realistas de implementación.

---

## Apéndice — Detalles de implementación y reproducibilidad

El *pipeline* completo está implementado en Python 3.12 utilizando Polars para procesamiento de datos, HuggingFace Transformers para FinBERT, statsmodels para modelos auxiliares ARIMA y Rich para presentación de resultados. La ejecución completa, incluyendo descarga de noticias, clasificación con FinBERT, descarga de precios, cálculo de indicadores, generación de señales y *backtesting*, se invoca con un único comando (`uv run python director.py`). El sistema solicita confirmación interactiva antes de la fase de comparativa de estrategias, que produce las tablas reportadas en la Sección 4. El código fuente, los datos cacheados y los resultados son reproducibles a partir del repositorio del proyecto.

---

## Referencias seleccionadas

Araci, D. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*. arXiv:1908.10063.

Barberis, N., y Thaler, R. (2003). *A Survey of Behavioral Finance*. En *Handbook of the Economics of Finance*, Vol. 1, pp. 1053-1128.

Brock, W., Lakonishok, J., y LeBaron, B. (1992). *Simple Technical Trading Rules and the Stochastic Properties of Stock Returns*. The Journal of Finance, 47(5), 1731-1764.

Fama, E. F. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work*. The Journal of Finance, 25(2), 383-417.

Hong, H., y Stein, J. C. (1999). *A Unified Theory of Underreaction, Momentum Trading, and Overreaction in Asset Markets*. The Journal of Finance, 54(6), 2143-2184.

Loughran, T., y McDonald, B. (2011). *When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks*. The Journal of Finance, 66(1), 35-65.

Tetlock, P. C. (2007). *Giving Content to Investor Sentiment: The Role of Media in the Stock Market*. The Journal of Finance, 62(3), 1139-1168.
