import numpy as np
from colorstreak import Logger


class TeoriaJuegos:
    """Juego de 2 jugadores: Señal Fundamental (sentimiento) vs Señal Técnica (promedios móviles).
    Estrategias: COMPRA, HOLD, VENTA.
    Pagos: rendimientos reales promedio cuando cada combinación de estrategias ocurrió."""

    ESTRATEGIAS = ["COMPRA", "HOLD", "VENTA"]

    def __init__(self, rendimientos: np.ndarray, senales_tecnicas: list[str], senales_sentimiento: list[str]) -> None:
        self._rendimientos = rendimientos
        self._senales_tecnicas = senales_tecnicas
        self._senales_sentimiento = senales_sentimiento
        self._matriz_pagos: np.ndarray | None = None
        self._equilibrio: dict | None = None
        self._construir()

    def _construir(self) -> None:
        """Construye la matriz de pagos y encuentra el equilibrio de Nash."""
        n = min(len(self._rendimientos), len(self._senales_tecnicas), len(self._senales_sentimiento))
        if n == 0:
            Logger.warning("TeoriaJuegos: sin datos para construir la matriz")
            return

        rend = self._rendimientos[:n]
        tec = self._senales_tecnicas[:n]
        sent = self._senales_sentimiento[:n]

        # Matriz de pagos 3x3: filas = sentimiento, columnas = técnico
        matriz = np.zeros((3, 3))
        conteo = np.zeros((3, 3))

        idx = {s: i for i, s in enumerate(self.ESTRATEGIAS)}

        for r, s_tec, s_sent in zip(rend, tec, sent):
            i = idx.get(s_sent)
            j = idx.get(s_tec)
            if i is not None and j is not None:
                matriz[i][j] += r
                conteo[i][j] += 1

        # Promediar pagos (evitar div/0)
        with np.errstate(divide="ignore", invalid="ignore"):
            matriz = np.where(conteo > 0, matriz / conteo, 0.0)

        self._matriz_pagos = matriz
        self._equilibrio = self._encontrar_nash(matriz)

    def _encontrar_nash(self, matriz: np.ndarray) -> dict:
        """Encuentra equilibrio de Nash en estrategias puras (mejor respuesta mutua)."""
        filas, cols = matriz.shape

        # Mejor respuesta del jugador fila (sentimiento) para cada columna
        mejor_fila = np.argmax(matriz, axis=0)  # para cada col, qué fila es mejor
        # Mejor respuesta del jugador columna (técnico) para cada fila
        mejor_col = np.argmax(matriz, axis=1)  # para cada fila, qué col es mejor

        equilibrios = []
        for i in range(filas):
            for j in range(cols):
                if mejor_fila[j] == i and mejor_col[i] == j:
                    equilibrios.append({
                        "sentimiento": self.ESTRATEGIAS[i],
                        "tecnico": self.ESTRATEGIAS[j],
                        "pago": round(float(matriz[i][j]), 6),
                    })

        if not equilibrios:
            # Sin equilibrio puro → estrategia mixta: elegir la combinación con mayor pago
            i, j = np.unravel_index(np.argmax(matriz), matriz.shape)
            equilibrios.append({
                "sentimiento": self.ESTRATEGIAS[i],
                "tecnico": self.ESTRATEGIAS[j],
                "pago": round(float(matriz[i][j]), 6),
                "tipo": "mejor_respuesta_global",
            })

        return {"equilibrios": equilibrios}

    @property
    def matriz_pagos(self) -> np.ndarray | None:
        return self._matriz_pagos

    @property
    def resumen(self) -> dict:
        """Resumen: matriz de pagos formateada y equilibrios encontrados."""
        if self._matriz_pagos is None:
            return {"construido": False}

        matriz_dict = {}
        for i, sent in enumerate(self.ESTRATEGIAS):
            for j, tec in enumerate(self.ESTRATEGIAS):
                matriz_dict[f"({sent},{tec})"] = round(float(self._matriz_pagos[i][j]), 6)

        return {
            "construido": True,
            "matriz_pagos": matriz_dict,
            "equilibrio_nash": self._equilibrio,
        }
