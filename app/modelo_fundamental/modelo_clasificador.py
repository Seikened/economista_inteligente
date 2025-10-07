from functools import lru_cache
from transformers import pipeline



# ===============================================================================================

@lru_cache(maxsize=1)
def clasificador_cache():
    return pipeline(
        task="text-classification",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert",
        top_k=None,
        truncation=True,            # corta seguro a 512 tokens
        max_length=512,
        )


class ClasificadorSentimientos:
    
    def __init__(self,noticia : dict):
        self.noticia = noticia
        self.clasificador = clasificador_cache()
        self.probabilidad_positiva = 0.0
        self.probabilidad_negativa = 0.0
        self.probabilidad_neutra = 0.0
        self.sentimiento = ""


    def _ordenar_probabilidades(self, resultados):
        puntajes_ordenados = sorted(resultados, key=lambda x: x["score"], reverse=True)
        etiqueta_principal = puntajes_ordenados[0]
        suma_total = sum(p["score"] for p in resultados) or 1.0
        probabilidades = {p["label"]: p["score"]/suma_total for p in resultados}
        return etiqueta_principal["label"], {k: round(v, 4) for k, v in probabilidades.items()}


    def _clasificar_titulo_noticia(self):
        titulo = self.noticia.get("headline", "")
        resultados = self.clasificador(titulo)

        puntajes = resultados[0] 
        etiqueta_principal, probabilidades = self._ordenar_probabilidades(puntajes)

        resultado = {
            "sent": etiqueta_principal,
            "positive": probabilidades.get("positive", 0.0),
            "negative": probabilidades.get("negative", 0.0),
            "neutral": probabilidades.get("neutral", 0.0)
        }
        return resultado


    def _clasificar_resumen_noticia(self):
        resumen = self.noticia.get("summary", "")
        tamaño_noticia = len(resumen)
        
        # Si viene vacío o solo espacios, sal temprano con neutro
        if not resumen.strip():
            return {"sent": "neutral", "positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        chunks = []
        if tamaño_noticia > 512:
            chunks = [resumen[i:i + 512] for i in range(0, len(resumen), 512)]
        else:
            chunks = [resumen]
        
        resultados_lista = [self.clasificador(chunk) for chunk in chunks]
        positivos = []
        negativos = []
        neutros = []
        for resultados in resultados_lista:
            puntajes = resultados[0] 
            etiqueta_principal, probabilidades = self._ordenar_probabilidades(puntajes)
            positivos.append(probabilidades.get("positive", 0.0))
            negativos.append(probabilidades.get("negative", 0.0))
            neutros.append(probabilidades.get("neutral", 0.0))
        
        # Promedio pero por pesos de los chunks
        total_peso = sum(len(chunk) for chunk in chunks)
        pesos = [len(chunk) / total_peso for chunk in chunks]
        positivos = [p * peso for p, peso in zip(positivos, pesos)]
        negativos = [n * peso for n, peso in zip(negativos, pesos)]
        neutros = [ne * peso for ne, peso in zip(neutros, pesos)]
        
        # Promedio simple
        probabilidad_promedio = {
            "positive": round(sum(positivos), 4),
            "negative": round(sum(negativos), 4),
            "neutral": round(sum(neutros), 4)
        }
        
        etiqueta_principal = max(probabilidad_promedio, key=probabilidad_promedio.get) # type: ignore
        
        resultado = {
            "sent": etiqueta_principal,
            "positive": probabilidad_promedio["positive"],
            "negative": probabilidad_promedio["negative"],
            "neutral": probabilidad_promedio["neutral"]
        }
        return resultado
    
    
            
    
    
    def clasificar_noticia(self):
        resultado_titulo = self._clasificar_titulo_noticia()
        resultado_resumen = self._clasificar_resumen_noticia()
        peso_titulo = 0.3
        peso_resumen = 1 - peso_titulo
        self.probabilidad_positiva = round((resultado_titulo["positive"] * peso_titulo + resultado_resumen["positive"] * peso_resumen), 4)
        self.probabilidad_negativa = round((resultado_titulo["negative"] * peso_titulo + resultado_resumen["negative"] * peso_resumen), 4)
        self.probabilidad_neutra = round((resultado_titulo["neutral"] * peso_titulo + resultado_resumen["neutral"] * peso_resumen), 4)

        probabilidades = {
            "positive": self.probabilidad_positiva,
            "negative": self.probabilidad_negativa,
            "neutral": self.probabilidad_neutra
        }
        self.sentimiento = max(probabilidades, key=probabilidades.get) # type: ignore
        probabilidad_mas_alta = probabilidades[self.sentimiento]
        
        return self.sentimiento, probabilidad_mas_alta



    @classmethod
    def predecir(cls, noticia: dict):
        instancia = cls(noticia)
        return instancia.clasificar_noticia()
    
if __name__ == "__main__":
    noticia = {
        "title": "Título de la noticia",
        "summary": "Resumen de la noticia"
    }
    sentimiento, probabilidad = ClasificadorSentimientos.predecir(noticia)
    print(f"Sentimiento: {sentimiento}, Probabilidad: {probabilidad}")