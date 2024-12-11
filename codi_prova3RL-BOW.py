import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import mean_absolute_error

# Configurar lemmatizer
lemmatizer = WordNetLemmatizer()

# Funció per preprocessar el text
def preprocess_text(text):
    # Convertir a minúscules
    text = text.lower()
    # Tokenització manual: dividim el text per espais
    tokens = text.split()
    # Lematitzar cada paraula
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    return ' '.join(tokens)

# Funció per carregar i preprocessar les dades
def load_and_preprocess_data(path):
    # Carregar dataset
    dataset = pd.read_csv(path, usecols=['Id', 'review/score', 'review/text'])
    dataset = dataset.dropna()  # Eliminar files amb valors nuls
    
    # Agrupar els scores en categories
    def categorize_score(score):
        if score in [1, 2]:
            return 'Negatiu'
        elif score == 3:
            return 'Neutral'
        elif score in [4, 5]:
            return 'Positiu'
    
    dataset['category'] = dataset['review/score'].astype(int).apply(categorize_score)
    
    # Preprocessar el text
    dataset['processed_text'] = dataset['review/text'].apply(preprocess_text)
    return dataset

# Carregar dataset
dataset_path = 'reduced_reviews.csv'  # Canvia aquest path si cal
dataset = load_and_preprocess_data(dataset_path)

# Mostra un resum del dataset
print("Dimensionalitat del dataset:", dataset.shape)
print(dataset.head())

# Divisió del dataset en entrenament i test
X = dataset['processed_text']
y = dataset['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorització del text (Bag of Words)
vectorizer = CountVectorizer(max_features=5000)  # Limitem les característiques
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Entrenar el model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
model.fit(X_train_vec, y_train)

# Prediccions
y_pred = model.predict(X_test_vec)

# Avaluació del model
print("\nInforme de classificació:")
print(classification_report(y_test, y_pred))

# Visualització de la matriu de confusió
conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Matriu de Confusió")
plt.show()


# Càlcul del Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test.map({'Negatiu': -1, 'Neutral': 0, 'Positiu': 1}), 
                          pd.Series(y_pred).map({'Negatiu': -1, 'Neutral': 0, 'Positiu': 1}))

# Càlcul de l'error relatiu (proporció d'errors)
errors_relatius = sum(y_test != y_pred) / len(y_test)

print(f"\nMean Absolute Error (MAE): {mae:.2f}")
print(f"Error relatiu: {errors_relatius:.2%}")
