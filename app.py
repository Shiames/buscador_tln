from pathlib import Path
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import spacy
import streamlit as st

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec

nlp = spacy.load("es_core_news_sm")


def segmentar(texto):
    doc = nlp(texto)
    oraciones = [s.text.strip() for s in doc.sents]
    tokens = [t.text for t in doc if not t.is_space]
    return oraciones, tokens


def normalizar(texto):
    doc = nlp(texto.lower())
    tokens = []

    for token in doc:
        if token.is_space or token.is_punct:
            continue
        if token.is_stop and token.lemma_ != "no":
            continue
        if not token.is_alpha:
            continue
        tokens.append(token.lemma_)

    return " ".join(tokens)


def mostrar_pos(texto):
    doc = nlp(texto)
    for token in doc:
        print(token.text, token.pos_)


def bolsa_de_palabras(documents):
    vectorizador_bow = CountVectorizer()
    X_bow = vectorizador_bow.fit_transform(documents)
    print(vectorizador_bow.get_feature_names_out())
    print(X_bow.toarray())
    return vectorizador_bow, X_bow


def construir_indice(documentos_normalizados):
    vectorizador = TfidfVectorizer()
    X = vectorizador.fit_transform(documentos_normalizados)
    return vectorizador, X


def buscar(consulta, vectorizador, X):
    consulta_norm = normalizar(consulta)
    q = vectorizador.transform([consulta_norm])
    similitudes = cosine_similarity(q, X).flatten()
    return similitudes


def palabras_clave(vectorizador, fila, n=5):
    terminos = vectorizador.get_feature_names_out()
    valores = fila.toarray().flatten()
    orden = np.argsort(valores)[::-1]

    resultado = []
    for i in orden:
        if valores[i] > 0:
            resultado.append((terminos[i], valores[i]))
        if len(resultado) == n:
            break
    return resultado


def extraer_entidades(texto):
    doc = nlp(texto)
    return [(ent.text, ent.label_) for ent in doc.ents]


def analizar_dependencias(texto):
    doc = nlp(texto)
    return [
        (t.text, t.pos_, t.dep_, t.head.text)
        for t in doc
    ]


def extraer_patrones(texto):
    correos = re.findall(
        r"[\w\._]+@[\w\._]+\.\w+",
        texto
    )

    fechas = re.findall(
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        texto
    )

    urls = re.findall(
        r"https?://[^\s]+",
        texto
    )

    return correos, fechas, urls


def resumir(texto, n=2):
    doc = nlp(texto)
    oraciones = [s.text.strip() for s in doc.sents]

    textos = [normalizar(s) for s in oraciones]
    vectorizador = TfidfVectorizer()
    X = vectorizador.fit_transform(textos)

    puntajes = np.asarray(X.sum(axis=1)).ravel()
    indices = sorted(np.argsort(puntajes)[-n:])

    return [oraciones[i] for i in indices]


def responder(texto, pregunta):
    doc = nlp(texto)
    oraciones = [s.text.strip() for s in doc.sents]

    base = [normalizar(o) for o in oraciones]
    pregunta_norm = normalizar(pregunta)

    vectorizador = TfidfVectorizer()
    X = vectorizador.fit_transform(base + [pregunta_norm])

    similitudes = cosine_similarity(X[-1], X[:-1]).flatten()
    indice = similitudes.argmax()

    return oraciones[indice], similitudes[indice]


def asociaciones(tokens, ventana=2):
    cooc = defaultdict(int)

    for i, palabra in enumerate(tokens):
        inicio = max(0, i - ventana)
        fin = min(len(tokens), i + ventana + 1)

        for j in range(inicio, fin):
            if i != j:
                cooc[(palabra, tokens[j])] += 1

    return cooc


def entrenar_word2vec(documentos_normalizados):
    corpus = [
        texto.split()
        for texto in documentos_normalizados
    ]

    modelo = Word2Vec(
        sentences=corpus,
        vector_size=40,
        window=3,
        min_count=1,
        workers=1,
        seed=10
    )

    return modelo


st.set_page_config(
    page_title="MiniBuscador TLN",
    layout="wide"
)

st.title("MiniBuscador TLN")

archivos = st.file_uploader(
    "Sube archivos .txt",
    type=["txt"],
    accept_multiple_files=True
)

documentos = []
nombres = []

if archivos:
    for archivo in archivos:
        contenido = archivo.read().decode("utf-8", errors="ignore")
        documentos.append(contenido)
        nombres.append(archivo.name)
else:
    carpeta = Path("documents")
    if carpeta.exists():
        for ruta in sorted(carpeta.glob("*.txt")):
            documentos.append(ruta.read_text(encoding="utf-8"))
            nombres.append(ruta.name)

if not documentos:
    st.warning("No hay documentos. Sube archivos .txt o colocalos en la carpeta documents/")
    st.stop()

documentos_normalizados = [normalizar(d) for d in documentos]

vectorizador, X = construir_indice(documentos_normalizados)

consulta = st.text_input("Escribe una consulta")

if consulta:
    similitudes = buscar(consulta, vectorizador, X)
    resultados = sorted(
        zip(nombres, similitudes),
        key=lambda x: x[1],
        reverse=True
    )

    st.write("Resultados ordenados por similitud")
    df = pd.DataFrame(resultados, columns=["Documento", "Similitud"])
    st.dataframe(df, use_container_width=True)

    for nombre, sim in resultados:
        with st.expander(f"{nombre} - similitud: {sim:.4f}"):
            idx = nombres.index(nombre)
            st.write(documentos[idx])

st.header("Palabras clave")

opcion_clave = st.selectbox("Selecciona un documento", nombres, key="clave")
idx_clave = nombres.index(opcion_clave)
top = palabras_clave(vectorizador, X[idx_clave], n=5)
st.write(top)

st.header("Analisis linguistico")

opcion_ling = st.selectbox("Selecciona un documento", nombres, key="ling")
texto_ling = documentos[nombres.index(opcion_ling)]

st.subheader("Categorias gramaticales")
st.dataframe(
    pd.DataFrame(mostrar_pos_visible(texto_ling), columns=["Token", "POS"])
    if False else pd.DataFrame(
        [(t.text, t.pos_) for t in nlp(texto_ling)],
        columns=["Token", "POS"]
    ),
    use_container_width=True
)

st.subheader("Dependencias")
st.dataframe(
    pd.DataFrame(analizar_dependencias(texto_ling), columns=["Token", "POS", "Dep", "Head"]),
    use_container_width=True
)

st.subheader("Entidades")
st.dataframe(
    pd.DataFrame(extraer_entidades(texto_ling), columns=["Entidad", "Tipo"]),
    use_container_width=True
)

st.header("Datos extraidos")

for nombre, texto in zip(nombres, documentos):
    correos, fechas, urls = extraer_patrones(texto)
    st.write(nombre)
    st.write("Correos:", correos)
    st.write("Fechas:", fechas)
    st.write("URLs:", urls)

st.header("Resumen")

opcion_res = st.selectbox("Selecciona un documento", nombres, key="res")
n_res = st.slider("Numero de oraciones", 1, 5, 2)
resumen = resumir(documentos[nombres.index(opcion_res)], n=n_res)
st.write(resumen)

st.header("Preguntas")

opcion_preg = st.selectbox("Selecciona un documento", nombres, key="preg")
pregunta = st.text_input("Escribe una pregunta")
if pregunta:
    respuesta, sim = responder(documentos[nombres.index(opcion_preg)], pregunta)
    st.write("Respuesta:", respuesta)
    st.write("Similitud:", sim)

st.header("Palabras similares")

modelo = entrenar_word2vec(documentos_normalizados)
vocab = list(modelo.wv.index_to_key)
palabra = st.selectbox("Selecciona una palabra", vocab, key="w2v")
if palabra:
    similares = modelo.wv.most_similar(palabra, topn=5)
    st.write(similares)
