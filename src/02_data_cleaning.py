import pandas as pd
from scipy.io import arff


# ==========================================
# 1. LOAD DATASET
# ==========================================

data, meta = arff.loadarff("data/Rice_Cammeo_Osmancik.arff")

df = pd.DataFrame(data)

# Convert Class from bytes to string
df["Class"] = df["Class"].str.decode("utf-8")


# ==========================================
# 2. MISSING VALUE CHECK
# ==========================================

print("MISSING VALUES - 02_data_cleaning.py:21")
print("= - 02_data_cleaning.py:22" * 50)

print(df.isnull().sum())


# ==========================================
# 3. DUPLICATE CHECK
# ==========================================

print("\nDUPLICATE ROWS - 02_data_cleaning.py:31")
print("= - 02_data_cleaning.py:32" * 50)

duplicates = df.duplicated().sum()

print("Number of duplicate rows: - 02_data_cleaning.py:36", duplicates)


# ==========================================
# 4. DATASET INFORMATION
# ==========================================

print("\nDATASET INFORMATION - 02_data_cleaning.py:43")
print("= - 02_data_cleaning.py:44" * 50)

df.info()


# ==========================================
# 5. STATISTICAL SUMMARY
# ==========================================

print("\nSTATISTICAL SUMMARY - 02_data_cleaning.py:53")
print("= - 02_data_cleaning.py:54" * 50)

print(df.describe())


# ==========================================
# 6. CLASS DISTRIBUTION
# ==========================================

print("\nCLASS DISTRIBUTION - 02_data_cleaning.py:63")
print("= - 02_data_cleaning.py:64" * 50)

print(df["Class - 02_data_cleaning.py:66"].value_counts())


# ==========================================
# 7. FINAL DATASET SHAPE
# ==========================================

print("\nFINAL DATASET SHAPE - 02_data_cleaning.py:73")
print("= - 02_data_cleaning.py:74" * 50)

print(df.shape)