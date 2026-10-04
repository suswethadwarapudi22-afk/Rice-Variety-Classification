import pandas as pd
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. LOAD DATASET
# ==========================================

data, meta = arff.loadarff(
    "data/Rice_Cammeo_Osmancik.arff"
)

df = pd.DataFrame(data)

df["Class"] = df["Class"].str.decode("utf-8")


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

X = df.drop("Class", axis=1)

y = df["Class"]


# ==========================================
# 3. ENCODE TARGET
# ==========================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)


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
# 5. STANDARDIZATION
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==========================================
# 6. DEFINE MODELS
# ==========================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf"
    )
}


# ==========================================
# 7. TRAIN AND EVALUATE
# ==========================================

results = []


for name, model in models.items():

    # Tree models can use original features.
    # Other models use scaled features.

    if name in ["Decision Tree", "Random Forest"]:

        model.fit(
            X_train,
            y_train
        )

        y_pred = model.predict(
            X_test
        )

    else:

        model.fit(
            X_train_scaled,
            y_train
        )

        y_pred = model.predict(
            X_test_scaled
        )


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


    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })


# ==========================================
# 8. CREATE RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(results)


print("\nMODEL COMPARISON - 06_model_comparison.py:181")
print("= - 06_model_comparison.py:182" * 80)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1 Score": "{:.4f}".format
        }
    )
)


# ==========================================
# 9. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)


print("\nResults saved to: - 06_model_comparison.py:207")
print("results/model_comparison.csv - 06_model_comparison.py:208")