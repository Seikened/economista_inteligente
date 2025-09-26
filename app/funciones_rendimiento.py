import numpy as np

# 1. Rendimientos simples por periodo
def rendimientos_simples(precios: np.array) -> np.array:
    return (precios[1:] - precios[:-1]) / precios[:-1]

# 2. Rendimientos logarítmicos por periodo
def rendimientos_log(precios: np.array) -> np.array:
    return np.log(precios[1:] / precios[:-1])

# 3. Rendimiento promedio aritmético
def rendimiento_promedio(precios: np.array) -> float:
    r_simple = rendimientos_simples(precios)
    return np.mean(r_simple)

# 4. Rendimiento anualizado
def rendimiento_anualizado(precios: np.array, periodos_por_anio: int) -> float:
    r_simple = rendimientos_simples(precios)
    r_acum = np.prod(1 + r_simple) - 1
    n_periodos = len(r_simple)
    return (1 + r_acum) ** (periodos_por_anio / n_periodos) - 1

# 5. Rendimiento real (ajustado por inflación)
def rendimiento_real(precios: np.array, inflacion: float) -> float:
    r_simple = rendimientos_simples(precios)
    return (1 + r_simple) / (1 + inflacion) - 1

def rendimiento_real_prueba(rendimiento_simple: np.array, inflacion: float) -> float:
    return (1 + rendimiento_simple) / (1 + inflacion) - 1

# 6. Rendimiento esperado
def rendimiento_esperado(rendimientos: np.array, probabilidades: np.array) -> float:
    return np.sum(rendimientos * probabilidades)