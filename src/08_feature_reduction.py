import pandas as pd

from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

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
# 2. DEFINE FEATURE RANKING
# ==========================================

feature_ranking = [
    "Perimeter",
    "Major_Axis_Length",
    "Area",
    "Convex_Area",
    "Eccentricity",
    "Minor_Axis_Length",
    "Extent"
]


# ==========================================
# 3. TARGET
# ==========================================

y = df["Class"]

encoder = LabelEncoder()

y = encoder.fit_transform(y)


# ==========================================
# 4. TEST DIFFERENT FEATURE COUNTS
# ==========================================

feature_counts = [7, 6, 5, 4, 3]

results = []


for count in feature_counts:

    selected_features = feature_ranking[:count]

    X = df[selected_features]


    # --------------------------------------
    # Train / Test Split
    # --------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # --------------------------------------
    # Scaling
    # --------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # --------------------------------------
    # Logistic Regression
    # --------------------------------------

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(
        X_train_scaled,
        y_train
    )


    # --------------------------------------
    # Prediction
    # --------------------------------------

    y_pred = model.predict(
        X_test_scaled
    )


    # --------------------------------------
    # Metrics
    # --------------------------------------

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


    # --------------------------------------
    # Store results
    # --------------------------------------

    results.append({

        "Feature_Count": count,

        "Features": ", ".join(
            selected_features
        ),

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1
    })


# ==========================================
# 5. RESULTS TABLE
# ==========================================

results_df = pd.DataFrame(
    results
)


print("\nFEATURE REDUCTION RESULTS")
print("=" * 100)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1_Score": "{:.4f}".format
        }
    )
)


# ==========================================
# 6. SAVE RESULTS
# ==========================================

results_df.to_csv(
    "results/feature_reduction_results.csv",
    index=False
)


print("\nResults saved to:")
print(
    "results/feature_reduction_results.csv"
)