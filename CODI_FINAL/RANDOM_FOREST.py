import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

##################################
# CARREGAR I PREPROCESSAR LES DADES
##################################

# Carregar dades
df_unbalanced = pd.read_csv('unbalanced_lemmatized.csv')
df_balanced = pd.read_csv('balanced_lemmatized.csv')
print("ok1")

# Preprocessament
for df in [df_unbalanced, df_balanced]:
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df['sentiment'] = df['score'].apply(lambda x: 'negatiu' if x <= 2 else 'neutre' if x == 3 else 'positiu')
    df['lemmatized_review'] = df['lemmatized_review'].str.replace(r'[^\w\s]', '', regex=True)
print("ok2")
# Vectorització
tfidf = TfidfVectorizer(max_features=100000, ngram_range=(1, 2))
X_unbalanced = tfidf.fit_transform(df_unbalanced['lemmatized_review'])
y_unbalanced = df_unbalanced['sentiment']
X_balanced = tfidf.transform(df_balanced['lemmatized_review'])
y_balanced = df_balanced['sentiment']
print("ok3")

##################################
# DIVISIÓ DELS DATASETS
##################################

X_train_ub, X_test_ub, y_train_ub, y_test_ub = train_test_split(X_unbalanced, y_unbalanced, test_size=0.2, random_state=42)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
print("ok4")

##################################
# ENTRENAMENT I AVALUACIÓ DEL MODEL
##################################

# Model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("ok5")

# Unbalanced Dataset
model.fit(X_train_ub, y_train_ub)
y_pred_ub = model.predict(X_test_ub)
accuracy_ub = accuracy_score(y_test_ub, y_pred_ub)
cm_ub = confusion_matrix(y_test_ub, y_pred_ub, labels=['negatiu', 'neutre', 'positiu'])
print("ok6")

# Balanced Dataset
model.fit(X_train_b, y_train_b)
y_pred_b = model.predict(X_test_b)
accuracy_b = accuracy_score(y_test_b, y_pred_b)
cm_b = confusion_matrix(y_test_b, y_pred_b, labels=['negatiu', 'neutre', 'positiu'])
print("ok7")

# Classification Reports
print("Random Forest - Unbalanced Dataset\n")
print(classification_report(y_test_ub, y_pred_ub))
print("Random Forest - Balanced Dataset\n")
print(classification_report(y_test_b, y_pred_b))

##################################
# MATRIUS DE CONFUSIÓ
##################################

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_ub, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=['negatiu', 'neutre', 'positiu'], yticklabels=['negatiu', 'neutre', 'positiu'])
axes[0].set_title('Random Forest - Unbalanced')
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Greens', ax=axes[1], xticklabels=['negatiu', 'neutre', 'positiu'], yticklabels=['negatiu', 'neutre', 'positiu'])
axes[1].set_title('Random Forest - Balanced')
plt.show()

##################################
# ROC CURVES
##################################

y_test_ub_bin = label_binarize(y_test_ub, classes=['negatiu', 'neutre', 'positiu'])
y_test_b_bin = label_binarize(y_test_b, classes=['negatiu', 'neutre', 'positiu'])
y_score_ub = model.predict_proba(X_test_ub)
y_score_b = model.predict_proba(X_test_b)

plt.figure(figsize=(10, 6))
for i, class_label in enumerate(['negatiu', 'neutre', 'positiu']):
    fpr_ub, tpr_ub, _ = roc_curve(y_test_ub_bin[:, i], y_score_ub[:, i])
    fpr_b, tpr_b, _ = roc_curve(y_test_b_bin[:, i], y_score_b[:, i])
    roc_auc_ub = auc(fpr_ub, tpr_ub)
    roc_auc_b = auc(fpr_b, tpr_b)
    plt.plot(fpr_ub, tpr_ub, linestyle='--', label=f'Unbalanced {class_label} (AUC={roc_auc_ub:.2f})')
    plt.plot(fpr_b, tpr_b, label=f'Balanced {class_label} (AUC={roc_auc_b:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid()
plt.show()

##################################
# ACCURACY VS PERCENTATGE DE DADES D'ENTRENAMENT
##################################

train_percentages = np.arange(10, 91, 10)  # Percentatges de 10% a 90%
accuracy_train_vs_test = {'unbalanced_train': [], 'unbalanced_test': [], 'balanced_train': [], 'balanced_test': []}

for train_size in train_percentages:
    split_ratio = train_size / 100.0
    X_train_ub, X_test_ub, y_train_ub, y_test_ub = train_test_split(X_unbalanced, y_unbalanced, train_size=split_ratio, random_state=42)
    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_balanced, y_balanced, train_size=split_ratio, random_state=42)
    
    model.fit(X_train_ub, y_train_ub)
    accuracy_train_vs_test['unbalanced_train'].append(accuracy_score(y_train_ub, model.predict(X_train_ub)))
    accuracy_train_vs_test['unbalanced_test'].append(accuracy_score(y_test_ub, model.predict(X_test_ub)))
    
    model.fit(X_train_b, y_train_b)
    accuracy_train_vs_test['balanced_train'].append(accuracy_score(y_train_b, model.predict(X_train_b)))
    accuracy_train_vs_test['balanced_test'].append(accuracy_score(y_test_b, model.predict(X_test_b)))

plt.figure(figsize=(10, 6))
plt.plot(train_percentages, accuracy_train_vs_test['unbalanced_train'], label='Unbalanced Train', linestyle='--')
plt.plot(train_percentages, accuracy_train_vs_test['unbalanced_test'], label='Unbalanced Test')
plt.plot(train_percentages, accuracy_train_vs_test['balanced_train'], label='Balanced Train', linestyle='--')
plt.plot(train_percentages, accuracy_train_vs_test['balanced_test'], label='Balanced Test')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Percentage of Training Data')
plt.legend()
plt.grid()
plt.show()

##################################
# EFECTE DEL NOMBRE D'ARBRES
##################################

n_estimators_values = [10, 50, 100, 200]
accuracy_n_estimators_analysis = {'unbalanced': [], 'balanced': []}

for n_estimators in n_estimators_values:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train_ub, y_train_ub)
    accuracy_n_estimators_analysis['unbalanced'].append(accuracy_score(y_test_ub, model.predict(X_test_ub)))
    model.fit(X_train_b, y_train_b)
    accuracy_n_estimators_analysis['balanced'].append(accuracy_score(y_test_b, model.predict(X_test_b)))

plt.figure(figsize=(8, 5))
plt.plot(n_estimators_values, accuracy_n_estimators_analysis['unbalanced'], marker='o', label='Unbalanced', linestyle='--')
plt.plot(n_estimators_values, accuracy_n_estimators_analysis['balanced'], marker='o', label='Balanced')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Accuracy')
plt.title('Effect of Number of Trees on Accuracy')
plt.legend()
plt.grid()
plt.show()
