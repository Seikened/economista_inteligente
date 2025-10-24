import json
import pandas as pd
import numpy as np
import os
import pickle
from arch import arch_model
import yfinance as yf

# === CONFIGURACIÓN ===
json_path = "noticias.json"  # tu dataset
model_dir = "garch_models"
start_date = "2024-06-01"
end_date = "2025-06-01"

os.makedirs(model_dir, exist_ok=True)

# === 1. Leer los tickers desde el JSON ===
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

tickers = list(data.keys())
print(f"📊 Se encontraron {len(tickers)} tickers en el JSON:\n{tickers}")

# === 2. Entrenar un modelo GARCH por cada ticker ===
model_info = []

for ticker in tickers:
    print(f"\n📈 Entrenando modelo GARCH para {ticker}...")

    try:
        # Descargar precios históricos
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            print(f"⚠️ No se pudieron obtener datos de {ticker}, se omite.")
            continue

        # Calcular rendimientos porcentuales
        df["returns"] = 100 * df["Close"].pct_change().dropna()

        # Entrenar modelo GARCH(1,1)
        model = arch_model(df["returns"].dropna(), vol="EGARCH", p=1, q=1)
        res = model.fit(disp="off")

        # Guardar modelo entrenado
        model_path = os.path.join(model_dir, f"{ticker}_garch.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(res, f)

        # Predicción de volatilidad futura (1 día)
        forecast = res.forecast(horizon=1)
        vol_pred = np.sqrt(forecast.variance.values[-1, 0])

        # Guardar información resumida
        params = res.params
        model_info.append({
            "ticker": ticker,
            "omega": params.get("omega", np.nan),
            "alpha[1]": params.get("alpha[1]", np.nan),
            "beta[1]": params.get("beta[1]", np.nan),
            "loglikelihood": res.loglikelihood,
            "aic": res.aic,
            "bic": res.bic,
            "volatilidad_predicha(1d)": vol_pred,
            "Corr:": np.corrcoef(res.conditional_volatility, np.abs(res.resid))[0,1]
        })

        print(f"✅ Modelo de {ticker} guardado en: {model_path}")

    except Exception as e:
        print(f"❌ Error con {ticker}: {e}")

# === 3. Guardar resumen en CSV ===
df_summary = pd.DataFrame(model_info)
df_summary.to_csv(os.path.join(model_dir, "garch_summary.csv"), index=False)

print("\n📄 Resumen de modelos guardado en garch_models/garch_summary.csv")
print(df_summary)
