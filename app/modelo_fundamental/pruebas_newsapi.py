import requests

news_key = '52b90e5e856647bbbadd64ff2854b887'

url = f"https://newsapi.org/v2/everything?q=tesla&language=es&sortBy=publishedAt&apiKey={news_key}"

response = requests.get(url)
data = response.json()

for article in data["articles"][:]:
    print(article["title"], "-", article["publishedAt"])

print('*'*60)
print('GDELT'*20)

import requests
import polars as pl

list_empresas = ['Tesla','Apple','Microsft']

df_empresas = pl.DataFrame([])
# Consulta a la API de GDELT
url = "https://api.gdeltproject.org/api/v2/doc/doc"

for empresa in list_empresas:
    params = {
        "query": empresa,
        "mode": "ArtList",
        "maxrecords": 100,
        "format": "JSON"
    }
    
    
    res = requests.get(url, params=params).json()
    
    if 'articles' in res:
        df_empresas_singular = pl.DataFrame(res['articles'])
        #Agregar columna para saber de que empresa es la noticia
        df_empresas_singular = df_empresas.with_columns(pl.lit(empresa).alias('Empresa'))
        df_empresas = pl.concat([df_empresas,df_empresas_singular])


print(df_empresas)
