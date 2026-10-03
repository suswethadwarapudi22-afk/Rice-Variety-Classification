import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
data, meta = arff.loadarff("data/Rice_Cammeo_Osmancik.arff")
df = pd.DataFrame(data)

# Decode class
df["Class"] = df["Class"].str.decode("utf-8")

# Six selected features
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
y = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Final tuned model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        C=10,
        solver="liblinear",
        max_iter=2000,
        random_state=42
    ))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Classification report
print("\nCLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nCONFUSION MATRIX")
print("=" * 60)
print(cm)

# Number of errors
errors = y_test != y_pred

print("\nERROR ANALYSIS")
print("=" * 60)
print("Total test samples :", len(y_test))
print("Correct predictions:", sum(~errors))
print("Incorrect predictions:", sum(errors))

# Create error dataframe
error_df = X_test.copy()

error_df["Actual"] = encoder.inverse_transform(y_test)
error_df["Predicted"] = encoder.inverse_transform(y_pred)

error_df = error_df[error_df["Actual"] != error_df["Predicted"]]

# Save errors
error_df.to_csv(
    "results/misclassified_samples.csv",
    index=False
)

print("\nMisclassified samples saved to:")
print("results/misclassified_samples.csv")

# Save confusion matrix
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix - Tuned Logistic Regression")

plt.tight_layout()

plt.savefig(
    "results/tuned_confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nConfusion matrix saved to:")
print("results/tuned_confusion_matrix.png")