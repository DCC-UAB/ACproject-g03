import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Cargar los datos en chunks para manejar datasets grandes
chunksize = 100000  # Procesar en partes de 100,000 filas
chunks = pd.read_csv('Books_rating.csv', chunksize=chunksize)

# Procesar los datos por partes
dataframes = []
for chunk in chunks:
    chunk = chunk[['review/text', 'review/score']].dropna()  # Seleccionar columnas necesarias
    dataframes.append(chunk)

# Combinar todos los chunks
df = pd.concat(dataframes, ignore_index=True)

# Convertir las puntuaciones en categorías de sentimiento
def categorize_sentiment(score):
    if score >= 4:
        return 'positive'
    elif score == 3:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['review/score'].apply(categorize_sentiment)

# Dividir los datos en características y etiquetas
X = df['review/text']
y = df['sentiment']

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear un pipeline para vectorización y modelo
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))),  # Tfidf + bigramas
    ('model', MultinomialNB())
])

# Entrenamiento del modelo
pipeline.fit(X_train, y_train)

# Evaluación del modelo
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Validación cruzada para evaluación robusta
cross_val_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {cross_val_scores.mean():.4f}")

"""
              precision    recall  f1-score   support

    negative       0.82      0.23      0.36     70236
     neutral       0.59      0.01      0.01     50790
    positive       0.82      1.00      0.90    478973

    accuracy                           0.82    599999
   macro avg       0.75      0.41      0.42    599999
weighted avg       0.80      0.82      0.76    599999
"""