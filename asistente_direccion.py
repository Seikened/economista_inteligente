from sklearn.feature_extraction.text import TfidfVectorizer
import json
import polars as pl
from transformers import AutoTokenizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm  
from app.modelo_matematico import funciones_rendimiento
import clasificador_NLP
import numpy as np
import pandas as pd
import pickle

def agregar_feature_prediccion_json(ruta_json, vectorizer, clf, clf_r, salida_json="noticias_actualizadas.json"):
    """
    Lee un JSON con estructura por tickers y agrega:
    - Predicciones de subida/bajada
    - Indicadores financieros: volatilidad, Sharpe y Sortino
    """

    with open(ruta_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for ticker, contenido in tqdm(data.items(), desc="Procesando tickers"):
        nuevas_noticias = []
        df_ticker = contenido["data"]

        # === Extraer rendimientos del ticker ===
        rendimientos = np.array([n["performance"] for n in df_ticker if "performance" in n])

        if len(rendimientos) > 1:
            # Calcular métricas con tus funciones
            volatil, volatil_anual = funciones_rendimiento.volatilidad(rendimientos)
            sharpe_d, sharpe_a = funciones_rendimiento.ratio_sharpe(rendimientos)
            sortino_d, sortino_a = funciones_rendimiento.sortino(rendimientos)
        else:
            volatil = volatil_anual = sharpe_d = sharpe_a = sortino_d = sortino_a = None

        # === Recorrer cada noticia ===
        for n in df_ticker:
            texto = n["headline"] + " " + n["summary"]
            X_new = vectorizer.transform([texto])

            pred = clf.predict(X_new)[0]
            prob = clf_r.predict_proba(X_new)[0]

            # Agregar predicciones y métricas
            n["pred_subida"] = pred
            n["prob_baja"] = float(prob[0])
            n["prob_subida"] = float(prob[1])
            n["pred_subida_num"] = 1 if pred == "positive" else 0

            n["volatilidad_diaria"] = volatil
            n["volatilidad_anual"] = volatil_anual
            n["sharpe_diario"] = sharpe_d
            n["sharpe_anual"] = sharpe_a
            n["sortino_diario"] = sortino_d
            n["sortino_anual"] = sortino_a

            nuevas_noticias.append(n)

        # Guardar de vuelta
        data[ticker]["data"] = nuevas_noticias

    with open(salida_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Archivo actualizado guardado en: {salida_json}")



def agregar_volatilidad_garch_json(ruta_json, salida_json="noticias_actualizadas.json"):
    """
    Lee un JSON con estructura por ticker y agrega:
      - 'volatility': volatilidad histórica estimada por el modelo GARCH
      - 'volatility_pred': volatilidad predicha para el día siguiente

    Guarda el resultado en un nuevo archivo JSON.
    """
    # === 1. Cargar dataset de noticias ===
    with open(ruta_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    tickers = list(data.keys())

    # === 2. Recorrer tickers ===
    for ticker in tqdm(tickers, desc="Procesando tickers GARCH"):
        records = data[ticker]["data"]

        # Crear DataFrame
        df = pl.DataFrame(records)
        df = df.with_columns([
            pl.col("date").str.strptime(pl.Datetime, "%Y-%m-%d",strict=False)
        ]).sort("date")
        
        # Verificar existencia del modelo GARCH
        model_path = f"garch_models/{ticker}_garch.pkl"
        try:
            with open(model_path, "rb") as f:
                res = pickle.load(f)
            print(f" Modelo GARCH cargado: {ticker}")
        except FileNotFoundError:
            print(f" No se encontró el modelo GARCH para {ticker}. Se omite.")
            continue

        # === 3. Obtener volatilidad condicional (histórica) ===
        vol_series = res.conditional_volatility.values

        # Ajustar longitudes (por si el modelo se entrenó con menos datos)
        min_len = min(len(vol_series), df.height)
        df = df.tail(min_len)
        vol_series = vol_series[-min_len:]

        df = df.with_columns(pl.Series("volatility",vol_series))

        # === 4. Predecir volatilidad futura (1 día adelante) ===
        forecast = res.forecast(horizon=1)
        future_vol = np.sqrt(forecast.variance.values[-1][0])  # σ predicha del siguiente día
        
        # Crear nueva columna 'volatility_pred'
        vol_pred = np.append(vol_series[1:],future_vol)
        df = df.with_columns(pl.Series('volatility_pred',vol_pred))
        
        df = df.with_columns(pl.col("date").cast(pl.Utf8))
        data[ticker]['data'] = df.to_dicts()


    # === 6. Guardar el JSON actualizado ===
    with open(salida_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Archivo actualizado con volatilidad GARCH guardado en: {salida_json}")


if __name__ == "__main__":
    ruta = 'noticias.json'
    df = clasificador_NLP.lectura(ruta)
    vectorizer2 , clf , clf_r = clasificador_NLP.modelo_PLN(df)
    agregar_feature_prediccion_json("noticias.json", vectorizer2, clf, clf_r)
    agregar_volatilidad_garch_json("noticias_actualizadas.json")