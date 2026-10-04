import pandas as pd
import numpy as np

from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data, meta = arff.loadarff(
    "data/Rice_Cammeo_Osmancik.arff"
)

df = pd.DataFrame(data)

# Convert Class from bytes to string
df["Class"] = df["Class"].str.decode("utf-8")


# ==========================================
# 2. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop("Class", axis=1)

y = df["Class"]


# ==========================================
# 3. ENCODE TARGET
# ==========================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)


print("Class Mapping: - 05_baseline_model.py:52")
for class_name, encoded_value in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
):
    print(class_name, "= - 05_baseline_model.py:57", encoded_value)


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 6. BASELINE MODEL
# ==========================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# ==========================================
# 7. PREDICTION
# ==========================================

y_pred = model.predict(X_test_scaled)


# ==========================================
# 8. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\nBASELINE MODEL RESULTS - 05_baseline_model.py:134")
print("= - 05_baseline_model.py:135" * 60)

print(f"Accuracy  : {accuracy:.4f} - 05_baseline_model.py:137")
print(f"Precision : {precision:.4f} - 05_baseline_model.py:138")
print(f"Recall    : {recall:.4f} - 05_baseline_model.py:139")
print(f"F1 Score  : {f1:.4f} - 05_baseline_model.py:140")


# ==========================================
# 9. CLASSIFICATION REPORT
# ==========================================

print("\nCLASSIFICATION REPORT - 05_baseline_model.py:147")
print("= - 05_baseline_model.py:148" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nCONFUSION MATRIX - 05_baseline_model.py:168")
print("= - 05_baseline_model.py:169" * 60)

print(cm)