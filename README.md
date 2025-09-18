# Economista Inteligente

Este proyecto implementa un sistema de predicción financiera que integra análisis técnico y análisis fundamental para ofrecer una visión más completa del mercado.

## Propósito General

El sistema está diseñado para que un "administrador" (el cerebro del sistema) coordine dos fuentes de información:

- **Análisis técnico:** Basado en precios históricos, tendencias y patrones clásicos del mercado.
- **Análisis fundamental:** Extrae y procesa información textual relevante (noticias, reportes, comunicados) para evaluar el sentimiento y el contexto del mercado.

La integración de ambos enfoques permite generar predicciones más robustas, considerando tanto los datos cuantitativos como la narrativa del entorno financiero.

## Estructura del Proyecto

- `main.py`: Punto de entrada principal del sistema.
- `app/`: Contiene la lógica principal dividida en módulos.
    - `director.py`: Implementa el administrador/coordinador del sistema.
    - `modelo_lenguaje/`: Módulos para el análisis fundamental.
        - `analizador/`: Procesamiento y evaluación de textos.
        - `extractor/`: Extracción de información relevante de textos.
    - `modelo_matematico/`: Módulos para el análisis técnico.
- `pyproject.toml`, `uv.lock`: Configuración y dependencias del proyecto.
- `README.md`: Este archivo.

Cada subcarpeta contiene su propio README con detalles específicos de su propósito y funcionamiento.


## Autores
- Fernando Leon Franco (AI Engineer)
- Rodrigo Mendoza Rodríguez (AI Engineer)
- Juan Yael Vazquez Avelar (AI Engineer)
- Emilio Frausto Ortiz (AI Engineer)
- Gael Arturo Muñoz Delgado (AI Engineer)
