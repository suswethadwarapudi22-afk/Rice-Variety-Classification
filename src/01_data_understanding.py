import pandas as pd
from scipy.io import arff


# Load ARFF dataset
data, meta = arff.loadarff("data/Rice_Cammeo_Osmancik.arff")

# Convert ARFF data into pandas DataFrame
df = pd.DataFrame(data)

# Convert Class values from bytes to normal strings
df["Class"] = df["Class"].str.decode("utf-8")


# Display first 5 rows
print("FIRST 5 ROWS - 01_data_understanding.py:16")
print("= - 01_data_understanding.py:17" * 50)
print(df.head())


# Display dataset shape
print("\nDATASET SHAPE - 01_data_understanding.py:22")
print("= - 01_data_understanding.py:23" * 50)
print(df.shape)


# Display column names
print("\nCOLUMN NAMES - 01_data_understanding.py:28")
print("= - 01_data_understanding.py:29" * 50)
print(df.columns.tolist())


# Display data types
print("\nDATA TYPES - 01_data_understanding.py:34")
print("= - 01_data_understanding.py:35" * 50)
print(df.dtypes)


# Display class distribution
print("\nCLASS DISTRIBUTION - 01_data_understanding.py:40")
print("= - 01_data_understanding.py:41" * 50)
print(df["Class - 01_data_understanding.py:42"].value_counts())


# Display missing values
print("\nMISSING VALUES - 01_data_understanding.py:46")
print("= - 01_data_understanding.py:47" * 50)
print(df.isnull().sum())