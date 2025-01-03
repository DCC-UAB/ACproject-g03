# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 13:56:30 2024

@author: Usuario
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

# Carregar datasets
balanced_df = pd.read_csv('unbalanced_lemmatized.csv')

# Transformar 'score' en categories
def categorize_score(df):
    df['label'] = df['score'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))
    return df

balanced_df = categorize_score(balanced_df)

# Preparar dades
def prepare_data(df):
    X = df['lemmatized_review']  # Columna amb el text processat
    y = df['label']  # Etiquetes transformades
    return train_test_split(X, y, test_size=0.2, random_state=42)

balanced_X_train, balanced_X_test, balanced_y_train, balanced_y_test = prepare_data(balanced_df)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "LinearSVC": LinearSVC()
}

# Funció per comparar TF-IDF i BoW
def compare_vectorizations(X_train, X_test, y_train, y_test, metric='accuracy', max_features=10000):
    vectorizers = {
        "TF-IDF": TfidfVectorizer(max_features=max_features),
        "BoW": CountVectorizer(max_features=max_features)
    }
    
    results = {vec_name: {model_name: 0 for model_name in models.keys()} for vec_name in vectorizers.keys()}
    
    for vec_name, vectorizer in vectorizers.items():
        # Vectorització
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Entrenar i avaluar cada model
        for model_name, model in models.items():
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            if metric == 'accuracy':
                results[vec_name][model_name] = accuracy_score(y_test, y_pred)
            elif metric == 'f1':
                results[vec_name][model_name] = f1_score(y_test, y_pred, average='weighted')
            else:
                raise ValueError("Mètrica no suportada. Utilitza 'accuracy' o 'f1'.")
    return results

# Comparar TF-IDF i BoW amb accuracy
results_accuracy = compare_vectorizations(
    balanced_X_train, balanced_X_test, balanced_y_train, balanced_y_test, metric='accuracy', max_features=10000
)

# Mostrar resultats
def plot_comparison(results, title, metric_name):
    plt.figure(figsize=(10, 6))
    for vec_name, model_scores in results.items():
        plt.plot(model_scores.keys(), model_scores.values(), label=vec_name, marker='o')
    plt.title(title)
    plt.xlabel('Models')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

plot_comparison(results_accuracy, "Comparació de Vectoritzacions (TF-IDF vs BoW) Unbalanced", "Accuracy")
