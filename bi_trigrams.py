import pandas as pd
import nltk
nltk.download('punkt_tab')
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import string

# Cargar el dataset
def load_data(file_path):
    """Carga el dataset desde el archivo CSV."""
    return pd.read_csv(FILE_PATH)

# Preprocesar texto
def preprocess_text(text):
    """Preprocesa el texto eliminando puntuaciones y convirtiendo a minúsculas."""
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Generar n-gramas
def generate_ngrams(text, n):
    """Genera n-gramas a partir de un texto."""
    tokens = word_tokenize(text)
    return list(ngrams(tokens, n))

# Contar n-gramas más frecuentes
def get_top_ngrams(corpus, n, top_k=10):
    """Obtiene los n-gramas más frecuentes en el corpus."""
    all_ngrams = []
    for text in corpus:
        text = preprocess_text(text)
        all_ngrams.extend(generate_ngrams(text, n))
    
    ngram_counts = Counter(all_ngrams)
    return ngram_counts.most_common(top_k)

# Ejecutar análisis principal
def main(file_path, top_k=10):
    """Ejecuta el análisis de bigramas y trigramas."""
    # Cargar datos
    data = load_data(file_path)

    # Combinar las columnas de texto
    corpus = data['review/summary'].fillna('') + ' ' + data['review/text'].fillna('')

    # Bigramas
    print("Top Bigramas:")
    top_bigrams = get_top_ngrams(corpus, 2, top_k)
    for bigram, count in top_bigrams:
        print(f"{' '.join(bigram)}: {count}")

    # Trigramas
    print("\nTop Trigramas:")
    top_trigrams = get_top_ngrams(corpus, 3, top_k)
    for trigram, count in top_trigrams:
        print(f"{' '.join(trigram)}: {count}")

if __name__ == "__main__":
    # Ruta al archivo CSV
    FILE_PATH = "Books_rating.csv"  # Cambiar si el archivo tiene otro nombre o ruta

    # Número de bigramas y trigramas más comunes a mostrar
    TOP_K = 10

    main(FILE_PATH, TOP_K)