import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.metrics import r2_score


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
    plt.show()

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

