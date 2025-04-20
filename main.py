import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from scipy import stats
import os
import re
from sklearn.metrics import classification_report, confusion_matrix

# ✅ Function to sanitize file names
def sanitize_filename(filename):
    filename = filename.replace(" ", "_")
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    return filename

# ✅ Create a folder to save plots
output_folder = "plots"
os.makedirs(output_folder, exist_ok=True)

# Load dataset
df = pd.read_csv("Student_depression.csv")

# Drop nulls
df.dropna(inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split features and label
X = df.drop("Depression", axis=1)
y = df["Depression"]

# Remove outliers using Z-score
z_scores = np.abs(stats.zscore(X))
X = X[(z_scores < 3).all(axis=1)]
y = y.loc[X.index].reset_index(drop=True)
X.reset_index(drop=True, inplace=True)

# ===================== 🖼️ VISUALIZATIONS ======================

# 🔹 Boxplot per feature
for col in X.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=X[col])
    plt.title(f"Box Plot: {col}")
    plt.tight_layout()
    sanitized_col = sanitize_filename(col)
    plt.savefig(f"{output_folder}/boxplot_{sanitized_col}.png")
    plt.close()

# 🔹 Histogram per feature
for col in X.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(X[col], bins=20, edgecolor='black')
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    sanitized_col = sanitize_filename(col)
    plt.savefig(f"{output_folder}/histogram_{sanitized_col}.png")
    plt.close()

# 🔹 Scatter plots for first 3 feature combinations
for i in range(min(3, len(X.columns)-1)):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=X.iloc[:, i], y=X.iloc[:, i+1], hue=y)
    plt.title(f"Scatter: {X.columns[i]} vs {X.columns[i+1]}")
    plt.xlabel(X.columns[i])
    plt.ylabel(X.columns[i+1])
    plt.tight_layout()
    plt.savefig(f"{output_folder}/scatter_{sanitize_filename(X.columns[i])}_vs_{sanitize_filename(X.columns[i+1])}.png")
    plt.close()

# 🔹 Bar chart for class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Bar Chart: Depression Label Distribution")
plt.xlabel("Depression Level")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{output_folder}/bar_depression_distribution.png")
plt.close()

# ===================== 🤖 MODEL TRAINING ======================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm_model.predict(X_test))

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test))

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))

# ===================== 📊 MODEL COMPARISON ======================

models = ['SVM', 'Random Forest', 'XGBoost']
accuracies = [svm_acc, rf_acc, xgb_acc]

plt.figure(figsize=(8, 5))
sns.barplot(x=models, y=accuracies)
plt.ylim(0, 1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig(f"{output_folder}/model_accuracy_comparison.png")
plt.close()
# Evaluate models and print results

# SVM evaluation
svm_preds = svm_model.predict(X_test)
print("=== 📌 Support Vector Machine ===")
print(classification_report(y_test, svm_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_preds))
print("\n")
# Random Forest evaluation
rf_preds = rf_model.predict(X_test)
print("=== 📌 Random Forest ===")
print(f"Accuracy: {rf_acc}")
print(classification_report(y_test, rf_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_preds))
print("\n")
# XGBoost evaluation
xgb_preds = xgb_model.predict(X_test)
print("=== 📌 XGBoost ===")
print(f"Accuracy: {xgb_acc}")
print(classification_report(y_test, xgb_preds))
print("Confusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))