import pandas as pd
from scipy.io import arff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


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


print("FEATURES - 04_feature_preparation.py:31")
print("= - 04_feature_preparation.py:32" * 50)

print(X.columns.tolist())


print("\nTARGET CLASSES - 04_feature_preparation.py:37")
print("= - 04_feature_preparation.py:38" * 50)

print(y.value_counts())


# ==========================================
# 3. ENCODE TARGET
# ==========================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("\nCLASS ENCODING - 04_feature_preparation.py:52")
print("= - 04_feature_preparation.py:53" * 50)

for class_name, encoded_value in zip(
    label_encoder.classes_,
    label_encoder.transform(label_encoder.classes_)
):
    print(class_name, "= - 04_feature_preparation.py:59", encoded_value)


# ==========================================
# 4. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("\nTRAINING DATA - 04_feature_preparation.py:75")
print("= - 04_feature_preparation.py:76" * 50)

print("X_train: - 04_feature_preparation.py:78", X_train.shape)
print("y_train: - 04_feature_preparation.py:79", y_train.shape)


print("\nTESTING DATA - 04_feature_preparation.py:82")
print("= - 04_feature_preparation.py:83" * 50)

print("X_test: - 04_feature_preparation.py:85", X_test.shape)
print("y_test: - 04_feature_preparation.py:86", y_test.shape)


# ==========================================
# 5. FEATURE STANDARDIZATION
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


print("\nSCALED DATA - 04_feature_preparation.py:100")
print("= - 04_feature_preparation.py:101" * 50)

print("X_train_scaled: - 04_feature_preparation.py:103", X_train_scaled.shape)
print("X_test_scaled: - 04_feature_preparation.py:104", X_test_scaled.shape)


# ==========================================
# 6. CLASS DISTRIBUTION AFTER SPLIT
# ==========================================

print("\nTRAIN CLASS DISTRIBUTION - 04_feature_preparation.py:111")
print("= - 04_feature_preparation.py:112" * 50)

print(pd.Series(y_train).value_counts())


print("\nTEST CLASS DISTRIBUTION - 04_feature_preparation.py:117")
print("= - 04_feature_preparation.py:118" * 50)

print(pd.Series(y_test).value_counts())


print("\nFEATURE PREPARATION COMPLETE - 04_feature_preparation.py:123")