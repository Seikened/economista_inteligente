from app.modelo_fundamental import Empresa
from app.modelo_matematico.funciones_rendimiento import *
import numpy as np

def rendimiento_esperado(rendimientos: np.array) -> float:
    if len(rendimientos) == 0:
        return 0
    p = np.ones(len(rendimientos)) / len(rendimientos)
    return np.sum(rendimientos * p) * 100

def volatilidad_historica(rendimientos: np.array) -> float:
    return np.std(rendimientos, ddof=1)

empresa_nvda = Empresa("NVDA") 

rendimientos = empresa_nvda.rendimiento_simple
volatilidad = volatilidad_historica(rendimientos)
vol_anual = volatilidad * np.sqrt(252)

E = rendimiento_esperado(rendimientos)

print(f"Rendimiento esperado NVDA: {E:.2f}%")
print(f'Volatilidad historica: {volatilidad:.4f}')
print(f'Volatilidad anual: {vol_anual:.4f}')