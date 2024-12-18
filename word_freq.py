import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt
import nltk

# Descargar recursos de NLTK si es necesario
nltk.download('punkt')
nltk.download('stopwords')

def load_data(file_path):
    """Carga el dataset desde un archivo CSV."""
    return pd.read_csv(file_path)

def preprocess_text(text):
    """Limpia y tokeniza el texto."""
    if pd.isnull(text):
        return []
    text = text.lower()  # Convertir a minúsculas
    text = text.translate(str.maketrans('', '', string.punctuation))  # Eliminar puntuación
    tokens = word_tokenize(text)  # Tokenización
    return tokens

def remove_stopwords(tokens):
    """Elimina palabras vacías de una lista de tokens."""
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words and word.isalpha()]

def calculate_word_frequencies(data, column, top_k=10):
    """Calcula las palabras más frecuentes en una columna de texto."""
    if column not in data.columns:
        raise ValueError(f"La columna '{column}' no está en el dataset.")

    all_tokens = []
    for text in data[column].fillna(''):
        tokens = preprocess_text(text)
        tokens = remove_stopwords(tokens)
        all_tokens.extend(tokens)

    word_counts = Counter(all_tokens)
    return word_counts.most_common(top_k)

def plot_word_frequencies(word_freqs, title):
    """Genera un gráfico de barras para las palabras más frecuentes."""
    words, counts = zip(*word_freqs)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Palabras")
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main(file_path, column, top_k):
    """Ejecuta el análisis de frecuencias de palabras."""
    print("Cargando datos...")
    data = load_data(file_path)

    print("Calculando frecuencias de palabras...")
    word_freqs = calculate_word_frequencies(data, column, top_k)

    print("Palabras más frecuentes:")
    for word, freq in word_freqs:
        print(f"{word}: {freq}")

    plot_word_frequencies(word_freqs, f"Top {top_k} palabras en '{column}'")

if __name__ == "__main__":
    # Configuración del script
    FILE_PATH = "Books_rating_sample.csv"  # Ruta del archivo
    COLUMN = "review/text"  # Columna de texto a analizar
    TOP_K = 10  # Número de palabras más comunes a mostrar

    main(FILE_PATH, COLUMN, TOP_K)
