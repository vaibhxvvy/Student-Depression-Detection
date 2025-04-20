# Student Depression Detection 📊🧠

This project focuses on analyzing and detecting depression levels among students using machine learning classification models. The dataset is preprocessed, and three models are trained and evaluated:

- Support Vector Machine (SVM)
- Random Forest Classifier
- XGBoost Classifier

## 📁 Dataset
- CSV File: `Student_depression.csv`
- Target Column: `Depression`
- Categorical columns are label-encoded.
- Outliers removed using Z-score.

## ⚙️ Models & Metrics

### 📌 Support Vector Machine
```
Accuracy: 0.5900
Classification Report:
              precision    recall  f1-score   support

           0       0.00      0.00      0.00      2310
           1       0.59      1.00      0.74      3258

Confusion Matrix:
[[   0 2310]
 [   0 3258]]
```

### 📌 Random Forest
```
Accuracy: 0.8466
Classification Report:
              precision    recall  f1-score   support

           0       0.83      0.79      0.81      2310
           1       0.86      0.89      0.87      3258

Confusion Matrix:
[[1829  481]
 [ 373 2885]]
```

### 📌 XGBoost
```
Accuracy: 0.8385
Classification Report:
              precision    recall  f1-score   support

           0       0.82      0.79      0.80      2310
           1       0.85      0.87      0.86      3258

Confusion Matrix:
[[1819  491]
 [ 408 2850]]
```

## ✅ How to Run
1. Clone this repo or download the project.
2. Place your CSV file named `Student_depression.csv` in the project root.
3. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python main.py
   ```

---
You can find all the generated plots and graphs in the following folder:

[View Plots](./sdd/plots)

Made by Vaibhav Surthi