import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Cargar los datos (asegúrate de reemplazar 'ruta_a_tu_archivo.csv' con la ruta real)
df = pd.read_csv('Books_rating.csv')

# Filtrar columnas necesarias
df = df[['review/text', 'review/score']].dropna()

# Convertir las puntuaciones en categorías de sentimiento
def categorize_sentiment(score):
    if score >= 4:
        return 'positive'
    elif score == 3:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['review/score'].apply(categorize_sentiment)

# Preparar datos
X = df['review/text']
y = df['sentiment']

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorización con Bag of Words
vectorizer = CountVectorizer(max_features=5000, stop_words='english')  # Ajusta max_features según el dataset
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Modelo de clasificación
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predicción
y_pred = model.predict(X_test_vec)

# Reporte de resultados
print(classification_report(y_test, y_pred))



"""
              precision    recall  f1-score   support

    negative       0.48      0.64      0.55     70236
     neutral       0.29      0.28      0.29     50790
    positive       0.91      0.87      0.89    478973

    accuracy                           0.79    599999
   macro avg       0.56      0.60      0.57    599999
weighted avg       0.81      0.79      0.80    599999
"""
