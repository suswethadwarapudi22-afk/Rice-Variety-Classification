import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import arff

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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
# 5. RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# ==========================================
# 6. FEATURE IMPORTANCE
# ==========================================

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})


# Sort from highest to lowest
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


# ==========================================
# 7. PRINT RESULTS
# ==========================================

print("\nFEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance.to_string(
        index=False,
        formatters={
            "Importance": "{:.4f}".format
        }
    )
)


# ==========================================
# 8. SAVE RESULTS
# ==========================================

feature_importance.to_csv(
    "results/feature_importance.csv",
    index=False
)


# ==========================================
# 9. CREATE BAR CHART
# ==========================================

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title(
    "Random Forest Feature Importance"
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "results/feature_importance.png",
    dpi=300
)

plt.close()


print("\nFeature importance saved to:")
print("results/feature_importance.csv")

print("results/feature_importance.png")