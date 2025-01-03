import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

# Carregar datasets
balanced_df = pd.read_csv('balanced_lemmatized.csv')
unbalanced_df = pd.read_csv('unbalanced_lemmatized.csv')

# Transformar 'score' en categories
def categorize_score(df):
    df['label'] = df['score'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))
    return df

balanced_df = categorize_score(balanced_df)
unbalanced_df = categorize_score(unbalanced_df)

# Preparar dades
def prepare_data(df):
    X = df['lemmatized_review']  # Columna amb el text processat
    y = df['label']  # Etiquetes transformades
    return train_test_split(X, y, test_size=0.2, random_state=42)

balanced_X_train, balanced_X_test, balanced_y_train, balanced_y_test = prepare_data(balanced_df)
unbalanced_X_train, unbalanced_X_test, unbalanced_y_train, unbalanced_y_test = prepare_data(unbalanced_df)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "LinearSVC": LinearSVC()
}

# Valors de max_features per comparar
max_features_values = [100, 500, 1000, 5000, 10000, 50000]

# Funció per entrenar i avaluar models
def evaluate_models(X_train, X_test, y_train, y_test):
    results = {model_name: [] for model_name in models.keys()}
    for max_features in max_features_values:
        # Vectorització
        vectorizer = TfidfVectorizer(max_features=max_features)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Entrenar i avaluar cada model
        for model_name, model in models.items():
            model.fit(X_train_tfidf, y_train)
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)   # Mètrica principal
            results[model_name].append(accuracy)
    return results

# Avaluar amb datasets balancejat i no balancejat
balanced_results = evaluate_models(balanced_X_train, balanced_X_test, balanced_y_train, balanced_y_test)
unbalanced_results = evaluate_models(unbalanced_X_train, unbalanced_X_test, unbalanced_y_train, unbalanced_y_test)

# Gràfics comparatius
def plot_results(results, title):
    plt.figure(figsize=(10, 6))
    for model_name, scores in results.items():
        plt.plot(max_features_values, scores, label=model_name, marker='o')
    plt.title(title)
    plt.xlabel('Max Features')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_results(balanced_results, "Rendiment segons max_features (Dataset Balancejat)")
plot_results(unbalanced_results, "Rendiment segons max_features (Dataset No Balancejat)")
