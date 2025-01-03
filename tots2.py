import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Carregar dades
df_unbalanced = pd.read_csv('unbalanced_lemmatized.csv')
df_balanced = pd.read_csv('balanced_lemmatized.csv')

# Preprocessament
for df in [df_unbalanced, df_balanced]:
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['sentiment'] = df['score'].apply(lambda x: 'negatiu' if x <= 2 else 'neutre' if x == 3 else 'positiu')
    df['lemmatized_review'] = df['lemmatized_review'].str.replace(r'[^\w\s]', '', regex=True)

# Vectorització
tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1, 2))
X_unbalanced = tfidf.fit_transform(df_unbalanced['lemmatized_review'])
y_unbalanced = df_unbalanced['sentiment']
X_balanced = tfidf.transform(df_balanced['lemmatized_review'])
y_balanced = df_balanced['sentiment']



# Divisió de dades
X_train_ub, X_test_ub, y_train_ub, y_test_ub = train_test_split(X_unbalanced, y_unbalanced, test_size=0.2, random_state=42)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Diccionaris per guardar precisions
accuracies_unbalanced = {}
accuracies_balanced = {}

###########################################
# Logistic Regression
###########################################

model = LogisticRegression(max_iter=1000, C=10)

# Unbalanced
model.fit(X_train_ub, y_train_ub)
y_pred_ub = model.predict(X_test_ub)
accuracy_ub = accuracy_score(y_test_ub, y_pred_ub)
cm_ub = confusion_matrix(y_test_ub, y_pred_ub, labels=['negatiu', 'neutre', 'positiu'])
print("Logistic Regression - Unbalanced Dataset\n")
print(classification_report(y_test_ub, y_pred_ub))
accuracies_unbalanced['Logistic Regression'] = accuracy_ub

# Balanced
model.fit(X_train_b, y_train_b)
y_pred_b = model.predict(X_test_b)
accuracy_b = accuracy_score(y_test_b, y_pred_b)
cm_b = confusion_matrix(y_test_b, y_pred_b, labels=['negatiu', 'neutre', 'positiu'])
print("Logistic Regression - Balanced Dataset\n")
print(classification_report(y_test_b, y_pred_b))
accuracies_balanced['Logistic Regression'] = accuracy_b

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_ub, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['negatiu', 'neutre', 'positiu'], yticklabels=['negatiu', 'neutre', 'positiu'])
axes[0].set_title('Logistic Regression - Unbalanced')
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=['negatiu', 'neutre', 'positiu'], yticklabels=['negatiu', 'neutre', 'positiu'])
axes[1].set_title('Logistic Regression - Balanced')
plt.show()

###########################################
# Decision Tree
###########################################

model = DecisionTreeClassifier(random_state=42)

# Unbalanced
model.fit(X_train_ub, y_train_ub)
y_pred_ub = model.predict(X_test_ub)
accuracy_ub = accuracy_score(y_test_ub, y_pred_ub)
cm_ub = confusion_matrix(y_test_ub, y_pred_ub, labels=['negatiu', 'neutre', 'positiu'])
print("Decision Tree - Unbalanced Dataset\n")
print(classification_report(y_test_ub, y_pred_ub))
accuracies_unbalanced['Decision Tree'] = accuracy_ub

# Balanced
model.fit(X_train_b, y_train_b)
y_pred_b = model.predict(X_test_b)
accuracy_b = accuracy_score(y_test_b, y_pred_b)
cm_b = confusion_matrix(y_test_b, y_pred_b, labels=['negatiu', 'neutre', 'positiu'])
print("Decision Tree - Balanced Dataset\n")
print(classification_report(y_test_b, y_pred_b))
accuracies_balanced['Decision Tree'] = accuracy_b

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_ub, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['negatiu', 'neutre', 'positiu'], yticklabels=['negatiu', 'neutre', 'positiu'])
axes[0].set_title('Decision Tree - Unbalanced')
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=['negatiu', 'neutre', 'positiu'], yticklabels=['negatiu', 'neutre', 'positiu'])
axes[1].set_title('Decision Tree - Balanced')
plt.show()

###########################################
# Naive Bayes
###########################################

model = MultinomialNB()

# Unbalanced
model.fit(X_train_ub, y_train_ub)
y_pred_ub = model.predict(X_test_ub)
accuracy_ub = accuracy_score(y_test_ub, y_pred_ub)
cm_ub = confusion_matrix(y_test_ub, y_pred_ub, labels=['negatiu', 'neutre', 'positiu'])
print("Naive Bayes - Unbalanced Dataset\n")
print(classification_report(y_test_ub, y_pred_ub))
accuracies_unbalanced['Naive Bayes'] = accuracy_ub

# Balanced
model.fit(X_train_b, y_train_b)
y_pred_b = model.predict(X_test_b)
accuracy_b = accuracy_score(y_test_b, y_pred_b)
cm_b = confusion_matrix(y_test_b, y_pred_b, labels=['negatiu', 'neutre', 'positiu'])
print("Naive Bayes - Balanced Dataset\n")
print(classification_report(y_test_b, y_pred_b))
accuracies_balanced['Naive Bayes'] = accuracy_b

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_ub, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['negatiu', 'neutre', 'positiu'], yticklabels=['negatiu', 'neutre', 'positiu'])
axes[0].set_title('Naive Bayes - Unbalanced')
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=['negatiu', 'neutre', 'positiu'], yticklabels=['negatiu', 'neutre', 'positiu'])
axes[1].set_title('Naive Bayes - Balanced')
plt.show()

###########################################
# Linear SVC
###########################################

model = LinearSVC(C=1, max_iter=1000)

# Unbalanced
model.fit(X_train_ub, y_train_ub)
y_pred_ub = model.predict(X_test_ub)
accuracy_ub = accuracy_score(y_test_ub, y_pred_ub)
cm_ub = confusion_matrix(y_test_ub, y_pred_ub, labels=['negatiu', 'neutre', 'positiu'])
print("Linear SVC - Unbalanced Dataset\n")
print(classification_report(y_test_ub, y_pred_ub))
accuracies_unbalanced['Linear SVC'] = accuracy_ub

# Balanced
model.fit(X_train_b, y_train_b)
y_pred_b = model.predict(X_test_b)
accuracy_b = accuracy_score(y_test_b, y_pred_b)
cm_b = confusion_matrix(y_test_b, y_pred_b, labels=['negatiu', 'neutre', 'positiu'])
print("Linear SVC - Balanced Dataset\n")
print(classification_report(y_test_b, y_pred_b))
accuracies_balanced['Linear SVC'] = accuracy_b

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_ub, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['negatiu', 'neutre', 'positiu'], yticklabels=['negatiu', 'neutre', 'positiu'])
axes[0].set_title('Linear SVC - Unbalanced')
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=['negatiu', 'neutre', 'positiu'], yticklabels=['negatiu', 'neutre', 'positiu'])
axes[1].set_title('Linear SVC - Balanced')
plt.show()

###########################################
# Comparació final
###########################################

# Comparació d'accuracies
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
df_acc = pd.DataFrame({'Model': accuracies_unbalanced.keys(),
                       'Unbalanced': accuracies_unbalanced.values(),
                       'Balanced': accuracies_balanced.values()})
df_acc.plot(x='Model', kind='bar', ax=ax)
plt.title('Accuracies Comparison')
plt.ylabel('Accuracy')
plt.show()
