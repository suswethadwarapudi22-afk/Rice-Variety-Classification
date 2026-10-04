import pandas as pd

from scipy.io import arff

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
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
# 2. USE REDUCED 6-FEATURE SET
# ==========================================

selected_features = [
    "Perimeter",
    "Major_Axis_Length",
    "Area",
    "Convex_Area",
    "Eccentricity",
    "Minor_Axis_Length"
]

X = df[selected_features]

y = df["Class"]


# ==========================================
# 3. ENCODE TARGET
# ==========================================

encoder = LabelEncoder()

y = encoder.fit_transform(y)


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
# 5. CREATE PIPELINE
# ==========================================

pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])


# ==========================================
# 6. HYPERPARAMETER GRID
# ==========================================

param_grid = {

    "model__C": [
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "model__solver": [
        "lbfgs",
        "liblinear"
    ]
}


# ==========================================
# 7. CROSS-VALIDATION
# ==========================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ==========================================
# 8. GRID SEARCH
# ==========================================

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1
)


print("Starting hyperparameter tuning... - 09_hyperparameter_tuning.py:147")

grid_search.fit(
    X_train,
    y_train
)


# ==========================================
# 9. BEST PARAMETERS
# ==========================================

print("\nBEST PARAMETERS - 09_hyperparameter_tuning.py:159")
print("= - 09_hyperparameter_tuning.py:160" * 60)

print(
    grid_search.best_params_
)


print("\nBEST CROSSVALIDATION F1 SCORE - 09_hyperparameter_tuning.py:167")
print("= - 09_hyperparameter_tuning.py:168" * 60)

print(
    f"{grid_search.best_score_:.4f}"
)


# ==========================================
# 10. TEST SET EVALUATION
# ==========================================

best_model = grid_search.best_estimator_

y_pred = best_model.predict(
    X_test
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


print("\nTUNED MODEL TEST RESULTS - 09_hyperparameter_tuning.py:210")
print("= - 09_hyperparameter_tuning.py:211" * 60)

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ==========================================
# 11. CLASSIFICATION REPORT
# ==========================================

print("\nCLASSIFICATION REPORT - 09_hyperparameter_tuning.py:234")
print("= - 09_hyperparameter_tuning.py:235" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)


# ==========================================
# 12. SAVE RESULTS
# ==========================================

results = pd.DataFrame({

    "Model": [
        "Tuned Logistic Regression"
    ],

    "Features": [
        6
    ],

    "Accuracy": [
        accuracy
    ],

    "Precision": [
        precision
    ],

    "Recall": [
        recall
    ],

    "F1_Score": [
        f1
    ]
})


results.to_csv(
    "results/tuned_model_results.csv",
    index=False
)


print("\nResults saved to: - 09_hyperparameter_tuning.py:284")
print(
    "results/tuned_model_results.csv"
)