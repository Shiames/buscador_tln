from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer as CV 
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from sklearn.metrics.pairwise import cosine_similarity as cs
import spacy
nlp=spacy.load("es_core_news_sm")

documentos=[
    "inteligancia artificial analiza texto",
    "inteligencia artificial procesa datos",
    "robot movil utiliza sensores" 
]

def segmentar(texto):
    doc = nlp(texto)
    oraciones = [s.text.strip() for s in doc.sents]
    tokens=[t.text for t in doc if not t.is_space]
    return oraciones, tokens
def normalizar(texto):
    doc = nlp(texto.lower())
    tokens=[]
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
        print("token.text, token.pos_")
        
vectorizer_bow=CV()
X_bow=vectorizer_bow.fit_transform(documentos)

print(vectorizer_bow.get_feature_names_out())
print(X_bow.toarray())

def construir_indice(documentos_normalizados):
    vectorizador=TFIDF()
    X=vectorizador.fit_transform(documentos_normalizados)   
    return vectorizador, X  

def buscar(consulta, vectorizador, X):
    consulta_norm=normalizar(consulta)
    q=vectorizador.transform([consulta_norm])
    similitudes=cs(q, X).flatten()
    return similitudes


        
