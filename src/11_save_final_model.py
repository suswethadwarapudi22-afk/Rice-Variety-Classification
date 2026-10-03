import os
import joblib
import pandas as pd

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# Load dataset
data, meta = arff.loadarff("data/Rice_Cammeo_Osmancik.arff")
df = pd.DataFrame(data)

# Decode class labels
df["Class"] = df["Class"].str.decode("utf-8")


# Selected lightweight features
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


# Encode target
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


# Final tuned Logistic Regression model
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=10,
        solver="liblinear",
        max_iter=2000,
        random_state=42
    ))
])


# Train final model
final_model.fit(X_train, y_train)


# Create models folder if necessary
os.makedirs("models", exist_ok=True)


# Save model
joblib.dump(
    final_model,
    "models/rice_classifier.joblib"
)


# Save label encoder
joblib.dump(
    encoder,
    "models/label_encoder.joblib"
)


# Save feature names
joblib.dump(
    selected_features,
    "models/selected_features.joblib"
)


print("FINAL MODEL SAVED - 11_save_final_model.py:90")
print("= - 11_save_final_model.py:91" * 60)

print("Model: - 11_save_final_model.py:93")
print("Logistic Regression - 11_save_final_model.py:94")

print("\nFeatures: - 11_save_final_model.py:96")
for feature in selected_features:
    print("", feature)

print("\nHyperparameters: - 11_save_final_model.py:100")
print("C = 10 - 11_save_final_model.py:101")
print("Solver = liblinear - 11_save_final_model.py:102")

print("\nSaved files: - 11_save_final_model.py:104")
print("models/rice_classifier.joblib - 11_save_final_model.py:105")
print("models/label_encoder.joblib - 11_save_final_model.py:106")
print("models/selected_features.joblib - 11_save_final_model.py:107")