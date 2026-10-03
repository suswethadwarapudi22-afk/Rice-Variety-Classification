import pandas as pd
from scipy.io import arff


# Load ARFF dataset
data, meta = arff.loadarff("data/Rice_Cammeo_Osmancik.arff")

# Convert ARFF data into pandas DataFrame
df = pd.DataFrame(data)

# Convert Class values from bytes to normal strings
df["Class"] = df["Class"].str.decode("utf-8")


# Display first 5 rows
print("FIRST 5 ROWS")
print("=" * 50)
print(df.head())


# Display dataset shape
print("\nDATASET SHAPE")
print("=" * 50)
print(df.shape)


# Display column names
print("\nCOLUMN NAMES")
print("=" * 50)
print(df.columns.tolist())


# Display data types
print("\nDATA TYPES")
print("=" * 50)
print(df.dtypes)


# Display class distribution
print("\nCLASS DISTRIBUTION")
print("=" * 50)
print(df["Class"].value_counts())


# Display missing values
print("\nMISSING VALUES")
print("=" * 50)
print(df.isnull().sum())