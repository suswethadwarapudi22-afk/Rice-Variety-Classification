import pandas as pd
import joblib
from scipy.io import arff

# Load dataset
data, meta = arff.loadarff("data/Rice_Cammeo_Osmancik.arff")
df = pd.DataFrame(data)

df["Class"] = df["Class"].str.decode("utf-8")

# Load model
model = joblib.load("models/rice_classifier.joblib")
encoder = joblib.load("models/label_encoder.joblib")

features = [
    "Perimeter",
    "Major_Axis_Length",
    "Area",
    "Convex_Area",
    "Eccentricity",
    "Minor_Axis_Length"
]

# Take actual samples from dataset
samples = df[features].sample(10, random_state=42)

predictions = model.predict(samples)
probabilities = model.predict_proba(samples)

print("\nTESTING MODEL WITH REAL DATASET SAMPLES - 12_test_predictions.py:30")
print("= - 12_test_predictions.py:31" * 70)

for i in range(len(samples)):
    predicted = encoder.inverse_transform([predictions[i]])[0]
    actual = df.loc[samples.index[i], "Class"]

    confidence = max(probabilities[i]) * 100

    print(f"\nSample {i + 1} - 12_test_predictions.py:39")
    print("" * 40)

    for feature in features:
        print(f"{feature:22}: {samples.iloc[i][feature]:.4f} - 12_test_predictions.py:43")

    print(f"Actual Class          : {actual} - 12_test_predictions.py:45")
    print(f"Predicted Class       : {predicted} - 12_test_predictions.py:46")
    print(f"Confidence            : {confidence:.2f}% - 12_test_predictions.py:47")