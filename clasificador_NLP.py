# ESTE ES EL CLASIFICADOR DE SUBIDAS O BAJADAS EN BASE AL HEADLINE Y SUMMARY DE UNA NOTICIA
#Basicamente predice si va a subir o bajar el rendimiento 
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import polars as pl
from transformers import AutoTokenizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import json
from tqdm import tqdm  


def evaluar_SVM(X, y, nombre,c):
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    clf = LinearSVC(C=c, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    clf_r = LogisticRegression(C=c, class_weight="balanced", max_iter=5000, solver="liblinear")
    clf_r.fit(X_train, y_train)
    y_pred_r = clf.predict(X_val)
    
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="macro")
    
    print(f"{nombre} → Accuracy: {acc:.3f}, F1-macro: {f1:.3f}")
    return nombre, acc, f1, clf , clf_r

def predecir_noticia(texto, vectorizer, clf, clf_r):
    # Transformar texto a vector
    X_new = vectorizer.transform([texto])
    # Predicción
    pred = clf.predict(X_new)[0]
    # Probabilidad aproximada (decision_function)
    decision = clf.decision_function(X_new)
    
    prob = clf_r.predict_proba(X_new)[0]
    return pred, decision, prob


def lectura(ruta):
    # 1. Leer JSON
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Convertir a lista de registros planos
    rows = []
    for ticker, contenido in data.items():
        noticias = contenido["data"]
        for n in noticias:
            n["ticker"] = ticker  # Agregamos la empresa como columna
            rows.append(n)

    # 3. Crear DataFrame
    df = pl.DataFrame(rows)
    return df


def modelo_PLN(df):
    c = 0.5
    # TF-IDF con n-gramas
    vectorizer2 = TfidfVectorizer(norm="l2", ngram_range=(1,3), min_df=5, stop_words="english")
    X = vectorizer2.fit_transform((df["headline"] + " " + df["summary"]).to_list())
    y = df['label']
    nombre, acc, f1 , clf, clf_r =evaluar_SVM(X, y, "TF-IDF 1-2gram filtrado",c)
    return vectorizer2,clf , clf_r



def main():
    ruta = 'noticias.json'
    df = lectura(ruta)
    vectorizer2 , clf , clf_r = modelo_PLN(df)
    
    
if __name__ == "__main__":
    main()