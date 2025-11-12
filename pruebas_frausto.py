import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# === 1. Cargar el JSON con tus datos ===
import json
with open("noticias_actualizadas.json", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for ticker, content in data.items():
    for item in content["data"]:
        item["ticker"] = ticker
        rows.append(item)

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["ticker", "date"]).dropna(subset=["performance"])

# === 2. Entrenar modelo para predecir performance ===
X = df[["prob_baja", "prob_subida", "pred_subida_num", "volatility_pred"]]
y = df["performance"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

df["performance_predicho"] = model.predict(X)

# === 3. Agregar medias por ticker ===
rend_ticker = df.groupby("ticker")["performance_predicho"].mean()
vol_ticker = df.groupby("ticker")["volatility_pred"].mean()

# === 4. Crear matriz de precios (agregada por día) ===
df_agg = (
    df.groupby(["date", "ticker"])["close"]
    .last()
    .reset_index()
)
pivot = df_agg.pivot(index="date", columns="ticker", values="close")

# Limpiar: eliminar columnas o filas con NaN
# En lugar de dropna, usa forward-fill y backward-fill
pivot = pivot.ffill().bfill()

# === 5. Calcular inputs ===
from pypfopt import risk_models, expected_returns

if pivot.empty or len(pivot.columns) < 2:
    raise ValueError("❌ No hay suficientes datos válidos para construir el portafolio.")

mu = df.groupby("ticker")["performance_predicho"].mean()
mu = mu.loc[pivot.columns]  # alinear tickers con pivot

S = risk_models.sample_cov(pivot)

# Limpieza adicional
mu = mu.replace([np.inf, -np.inf], np.nan).dropna()
S = S.replace([np.inf, -np.inf], np.nan).fillna(0)

# Alinear dimensiones
common_tickers = mu.index.intersection(S.columns)
mu = mu.loc[common_tickers]
S = S.loc[common_tickers, common_tickers]

if len(common_tickers) < 2:
    print("⚠️ Solo hay un ticker con datos válidos. No se puede optimizar portafolio.")
    print("Ticker dominante:", common_tickers[0])
    print("Rendimiento predicho:", mu.iloc[0])
else:
    # Asegurar que la matriz sea positiva definida
    S += np.eye(len(S)) * 1e-6

    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import objective_functions

    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=0.01)

    try:
        weights = ef.max_sharpe()
    except Exception as e:
        print("⚠️ Error en max_sharpe, se intentará min_volatility()")
        print(e)
        weights = ef.min_volatility()

    cleaned_weights = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance(verbose=True)

    print("\n=== Pesos óptimos del portafolio ===")
    for ticker, w in cleaned_weights.items():
        if w > 0:
            print(f"{ticker}: {w:.2%}")


# === Resultados finales ===
print("\n=== Desempeño esperado (escala diaria) ===")
print(f"Retorno esperado: {ret:.2%}")
print(f"Volatilidad esperada: {vol:.2%}")
print(f"Ratio de Sharpe: {sharpe:.2f}")

# === Escala anualizada ===
ret_anual = (1 + ret)**252 - 1
vol_anual = vol * np.sqrt(252)
sharpe_anual = (ret_anual / vol_anual) if vol_anual != 0 else np.nan

print("\n=== Desempeño esperado (anualizado) ===")
print(f"Retorno anualizado: {ret_anual:.2%}")
print(f"Volatilidad anualizada: {vol_anual:.2%}")
print(f"Sharpe anualizado: {sharpe_anual:.2f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(9,5))
plt.bar(weights.keys(), weights.values(), label="Peso óptimo")
plt.xticks(rotation=45)
plt.ylabel("Peso (%)")
plt.title("Pesos óptimos por activo (Máx. Sharpe)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
