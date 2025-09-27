from transformers import pipeline


class ClasificadorSentimientos:
    def __init__(self,noticia : dict):
        self.noticia = noticia
        self.clasificador = pipeline(
            task="text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            truncation=True,            # corta seguro a 512 tokens
            max_length=512,
        )
        
        
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
        resultados = self.clasificador(resumen)

        puntajes = resultados[0] 
        etiqueta_principal, probabilidades = self._ordenar_probabilidades(puntajes)

        resultado = {
            "sent": etiqueta_principal,
            "positive": probabilidades.get("positive", 0.0),
            "negative": probabilidades.get("negative", 0.0),
            "neutral": probabilidades.get("neutral", 0.0)
        }
        return resultado
    
    
    def clasificar_noticia(self):
        resultado_titulo = self._clasificar_titulo_noticia()
        resultado_resumen = self._clasificar_resumen_noticia()
        peso_titulo = 0.6
        peso_resumen = 0.4
        self.probabilidad_positiva = round((resultado_titulo["positive"] * peso_titulo + resultado_resumen["positive"] * peso_resumen), 4)
        self.probabilidad_negativa = round((resultado_titulo["negative"] * peso_titulo + resultado_resumen["negative"] * peso_resumen), 4)
        self.probabilidad_neutra = round((resultado_titulo["neutral"] * peso_titulo + resultado_resumen["neutral"] * peso_resumen), 4)

        probabilidades = {
            "positive": self.probabilidad_positiva,
            "negative": self.probabilidad_negativa,
            "neutral": self.probabilidad_neutra
        }
        self.sentimiento = max(probabilidades, key=probabilidades.get)
        probabilidad_mas_alta = probabilidades[self.sentimiento]
        
        return self.sentimiento, probabilidad_mas_alta


if __name__ == "__main__":

    noticia = {
        "headline": "Apple shares rose after the company announced stronger-than-expected iPhone sales.",
        "summary": "Apple's stock surged following the announcement of better-than-expected iPhone  sales, driven by strong demand in key markets."
    }

    clasificador = ClasificadorSentimientos(noticia)
    sentimiento, probabilidad = clasificador.clasificar_noticia()
    print(f"Sentimiento: {sentimiento}, Probabilidad: {probabilidad}")

