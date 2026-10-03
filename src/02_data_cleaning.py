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

print("MISSING VALUES")
print("=" * 50)

print(df.isnull().sum())


# ==========================================
# 3. DUPLICATE CHECK
# ==========================================

print("\nDUPLICATE ROWS")
print("=" * 50)

duplicates = df.duplicated().sum()

print("Number of duplicate rows:", duplicates)


# ==========================================
# 4. DATASET INFORMATION
# ==========================================

print("\nDATASET INFORMATION")
print("=" * 50)

df.info()


# ==========================================
# 5. STATISTICAL SUMMARY
# ==========================================

print("\nSTATISTICAL SUMMARY")
print("=" * 50)

print(df.describe())


# ==========================================
# 6. CLASS DISTRIBUTION
# ==========================================

print("\nCLASS DISTRIBUTION")
print("=" * 50)

print(df["Class"].value_counts())


# ==========================================
# 7. FINAL DATASET SHAPE
# ==========================================

print("\nFINAL DATASET SHAPE")
print("=" * 50)

print(df.shape)