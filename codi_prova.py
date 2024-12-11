# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:12:31 2024

@author: 04gao
"""

# Llibreries necessàries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Carregar el dataset
def load_dataset(path):
    dataset = pd.read_csv(path, header=0, delimiter=',')
    return dataset

# Carreguem el dataset (substitueix el path pel correcte)
dataset = load_dataset('Books_rating.csv')

# 2. Preprocessar les dades
# Creem una columna 'Sentiment' per categoritzar el review/score
def categorize_score(score):
    if score <= 2:
        return 'negatiu'
    elif score == 3:
        return 'neutral'
    else:
        return 'positiu'

# Afegir la columna de sentiment
dataset['Sentiment'] = dataset['review/score'].apply(categorize_score)

# Convertir les categories a valors numèrics
sentiment_mapping = {'negatiu': 0, 'neutral': 1, 'positiu': 2}
dataset['Sentiment'] = dataset['Sentiment'].map(sentiment_mapping)

# Mostrem les primeres files per validar
print(dataset[['review/score', 'Sentiment']].head())

# 3. Vectoritzar el text de les ressenyes
# Omplim valors nuls amb strings buits
dataset['review/text'] = dataset['review/text'].fillna('')

# Utilitzem TfidfVectorizer per vectoritzar el text
vectorizer = TfidfVectorizer(max_features=5000)  # Ajustem el nombre màxim de característiques
X_text = vectorizer.fit_transform(dataset['review/text'])

# Variables predictives (X) i variable objectiu (y)
X = X_text
y = dataset['Sentiment']

# Dividim en conjunts d'entrenament i test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entrenar el model de regressió logística multiclasse
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
model.fit(X_train, y_train)

# 5. Avaluar el model
y_pred = model.predict(X_test)

# Informe de classificació
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negatiu', 'Neutral', 'Positiu']))

# Matriu de confusió
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatiu', 'Neutral', 'Positiu'], yticklabels=['Negatiu', 'Neutral', 'Positiu'])
plt.xlabel('Predicció')
plt.ylabel('Real')
plt.title('Matriu de Confusió')
plt.show()

# 6. Visualització opcional de pesos del model
# Podem inspeccionar quines paraules tenen més pes per cada classe
feature_names = vectorizer.get_feature_names_out()
for i, class_label in enumerate(['Negatiu', 'Neutral', 'Positiu']):
    top_features = np.argsort(model.coef_[i])[-10:]  # Les 10 característiques amb més pes
    print(f"\nTop paraules per a la classe {class_label}:")
    print([feature_names[j] for j in top_features])
