import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import r2_score
from tqdm import tqdm

def agregar_volatilidades_a_json(ruta_json, salida_json="noticias_volatilidad.json"):
    """
    Agrega a cada noticia dentro del JSON:
    - 'volatility': volatilidad observada (GARCH)
    - 'volatility_pred': volatilidad predicha para el siguiente día

    Guarda el resultado en un nuevo JSON.
    """

    # === 1. Cargar JSON original ===
    with open(ruta_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # === 2. Recorrer tickers ===
    for ticker, contenido in tqdm(data.items(), desc="Añadiendo volatilidad y predicción"):
        try:
            # Convertir las noticias a DataFrame
            df = pd.DataFrame(contenido["data"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # === 3. Cargar modelo GARCH guardado ===
            model_path = f"garch_models/{ticker}_garch.pkl"
            with open(model_path, "rb") as f:
                res = pickle.load(f)

            # === 4. Obtener volatilidades condicionales ===
            vol_series = res.conditional_volatility.values

            # Ajustar longitudes automáticamente
            min_len = min(len(vol_series), len(df))
            df = df.tail(min_len).copy()
            vol_series = vol_series[-min_len:]
            df["volatility"] = vol_series

            # === 5. Obtener volatilidad predicha a 1 día ===
            forecast = res.forecast(horizon=1)
            future_vol = float(np.sqrt(forecast.variance.values[-1])[0])

            # === 6. Insertar resultados al JSON ===
            for i in range(len(df)):
                # Buscar noticia correspondiente (por fecha exacta)
                fecha = df.loc[i, "date"].strftime("%Y-%m-%d")
                for n in data[ticker]["data"]:
                    if n["date"] == fecha:
                        n["volatility"] = float(df.loc[i, "volatility"])
                        n["volatility_pred"] = future_vol
                        break

        except FileNotFoundError:
            print(f"⚠️ No se encontró modelo GARCH para {ticker}")
            continue
        except Exception as e:
            print(f"❌ Error procesando {ticker}: {e}")
            continue

    # === 7. Guardar JSON actualizado ===
    with open(salida_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Archivo actualizado guardado en: {salida_json}")

# === 1. Cargar dataset de noticias ===
with open("noticias.json", "r", encoding="utf-8") as f:
    data = json.load(f)

tickers = list(data.keys())

erres = []

for ticker in tickers:

    records = data[ticker]["data"]

    # === 3. Crear DataFrame ===
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # === 4. Rendimientos (performance * 100)
    returns = df["performance"] * 100

    # === 5. Cargar modelo GARCH guardado ===
    model_path = f"garch_models/{ticker}_garch.pkl"
    with open(model_path, "rb") as f:
        res = pickle.load(f)

    print(f"✅ Modelo GARCH cargado: {model_path}")

    vol_series = res.conditional_volatility.values

    # Ajustar longitudes automáticamente
    min_len = min(len(vol_series), len(df))
    df = df.tail(min_len).copy()
    vol_series = vol_series[-min_len:]

    df["volatility"] = vol_series



    # === 7. Tomar solo el último registro de cada día ===
    df_daily = df.groupby(df["date"].dt.date).tail(1)

    # === 8. Predecir volatilidad futura (ej. 3 días después) ===
    forecast = res.forecast(horizon=1)
    future_vols = np.sqrt(forecast.variance.values[-1])  # raíz cuadrada para σ

    # === 9. Crear nuevas fechas futuras ===
    last_date = df_daily["date"].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=len(future_vols) + 1, freq="D")[1:]

    # === 10. Graficar ===
    plt.figure(figsize=(10,5))
    plt.plot(df_daily["date"], df_daily["performance"]*100, label="Rendimiento (%)", color="gray", alpha=0.6)
    plt.plot(df_daily["date"], df_daily["volatility"], label="Volatilidad diaria (último valor)", color="red", linewidth=2)

    # Puntos de predicción futura
    plt.scatter(future_dates, future_vols, color="orange", s=30, label="Predicción futura (σ̂)")
    plt.plot(
        [df_daily["date"].iloc[-1]] + list(future_dates),
        [df_daily["volatility"].iloc[-1]] + list(future_vols),
        "--", color="orange", alpha=0.7
    )

    plt.title(f"GARCH - {ticker}\nPredicción de volatilidad futura")
    plt.xlabel("Fecha")
    plt.ylabel("Volatilidad (%)")
    plt.legend()
    plt.grid(True)
    #plt.show()

    # === 11. Mostrar predicciones futuras ===
    for i, vol in enumerate(future_vols, 1):
        print(f"Predicción de volatilidad para el día +{i}: {vol:.4f}")

    # === 12. Calcular R² del modelo de media condicional ===
    try:
        fitted_values = res.params.get("mu", 0) + res.resid / np.sqrt(res.conditional_volatility)
        corr = np.corrcoef(res.conditional_volatility, np.abs(res.resid))[0,1]
        erres.append(corr)
        #print(f"R² del modelo de media: {r2:.4f}")
    except Exception as e:
        print(f"No se pudo calcular R²: {e}")
        
print(np.mean(erres))
agregar_volatilidades_a_json("noticias_actualizadas.json")


