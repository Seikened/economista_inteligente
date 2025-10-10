from funciones_rendimiento import rendimiento_esperado
import numpy as np
probabilidades = np.array([0.3, 0.5, 0.2])
rendimientos = np.array([0.20, 0.10, -0.05])

rendimiento_esperado = rendimiento_esperado(rendimientos, probabilidades)
print(rendimiento_esperado)