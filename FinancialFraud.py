# Quantum-Inspired Financial Fraud Detection
# Author: Partha Sarathi

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score
from sklearn.utils import resample
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
data = pd.read_csv("creditcard.csv")  # make sure file is in same folder

print("✅ Data loaded successfully! Columns:", list(data.columns))
print("Total rows:", len(data))
print(data['Class'].value_counts())

# -------------------------------
# 2️⃣ Data Preprocessing (Handle imbalance safely)
# -------------------------------
fraud = data[data['Class'] == 1]
non_fraud = data[data['Class'] == 0]

if len(fraud) == 0:
    print("⚠️ No fraud samples found in dataset! Adding a few synthetic fraud rows for testing...")
    fraud = non_fraud.sample(n=50, random_state=42).copy()
    fraud['Class'] = 1

fraud_upsampled = resample(fraud,
                           replace=True,
                           n_samples=min(len(non_fraud)//2, len(fraud)*3),
                           random_state=42)

balanced_data = pd.concat([non_fraud, fraud_upsampled])

print("✅ Balanced dataset size:", len(balanced_data))
print(balanced_data['Class'].value_counts())

X = balanced_data.drop('Class', axis=1)
y = balanced_data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 3️⃣ Random Forest Model
# -------------------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_f1 = f1_score(y_test, rf_pred)
print("Random Forest F1-Score:", rf_f1)

# -------------------------------
# 4️⃣ Isolation Forest Model
# -------------------------------
iforest = IsolationForest(contamination=0.02, random_state=42)
iforest.fit(X_train)
if_pred = iforest.predict(X_test)
if_pred = np.where(if_pred == -1, 1, 0)
if_f1 = f1_score(y_test, if_pred)
print("Isolation Forest F1-Score:", if_f1)

# -------------------------------
# 5️⃣ Quantum-Inspired Optimization
# -------------------------------
def quantum_inspired_optimization(X, y, iterations=200):
    weights = np.random.randn(X.shape[1])
    best_weights = weights.copy()
    best_score = 0

    for i in range(iterations):
        new_weights = weights + np.random.normal(0, 0.05, size=weights.shape)
        y_pred = 1 / (1 + np.exp(-X.dot(new_weights)))  # sigmoid
        y_pred_class = (y_pred > 0.5).astype(int)
        score = f1_score(y, y_pred_class)

        delta = score - best_score
        if delta > 0 or np.exp(delta / 0.05) > np.random.rand():
            weights = new_weights
            best_score = score
            best_weights = weights.copy()

    return best_weights, best_score

q_weights, q_score = quantum_inspired_optimization(X_train, y_train)
print("Quantum-Inspired Model F1-Score:", q_score)

# -------------------------------
# 6️⃣ Compare Results
# -------------------------------
results = {
    "Random Forest": rf_f1,
    "Isolation Forest": if_f1,
    "Quantum-Inspired Model": q_score
}

plt.bar(results.keys(), results.values())
plt.title("Fraud Detection Model Comparison (F1-Score)")
plt.ylabel("F1-Score")
plt.show()

print("\nModel Performance Summary:")
for model, score in results.items():
    print(f"{model}: {round(score, 3)}")
