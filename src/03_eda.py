import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff


# ==========================================
# 1. LOAD DATASET
# ==========================================

data, meta = arff.loadarff("data/Rice_Cammeo_Osmancik.arff")

df = pd.DataFrame(data)

# Convert Class from bytes to string
df["Class"] = df["Class"].str.decode("utf-8")


# ==========================================
# 2. DEFINE FEATURES
# ==========================================

features = [
    "Area",
    "Perimeter",
    "Major_Axis_Length",
    "Minor_Axis_Length",
    "Eccentricity",
    "Convex_Area",
    "Extent"
]


# ==========================================
# 3. CLASS DISTRIBUTION
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Class"
)

plt.title("Rice Variety Class Distribution")
plt.xlabel("Rice Variety")
plt.ylabel("Number of Samples")

plt.tight_layout()

plt.savefig(
    "results/class_distribution.png",
    dpi=300
)

plt.close()


# ==========================================
# 4. FEATURE DISTRIBUTIONS
# ==========================================

df[features].hist(
    figsize=(14, 10),
    bins=30
)

plt.suptitle(
    "Distribution of Rice Morphological Features",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "results/feature_distributions.png",
    dpi=300
)

plt.close()


# ==========================================
# 5. BOXPLOTS
# ==========================================

plt.figure(figsize=(14, 8))

df[features].boxplot()

plt.title(
    "Boxplots of Rice Morphological Features"
)

plt.xticks(
    rotation=45
)

plt.ylabel("Feature Value")

plt.tight_layout()

plt.savefig(
    "results/feature_boxplots.png",
    dpi=300
)

plt.close()


# ==========================================
# 6. CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(10, 8))

correlation = df[features].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Correlation Between Rice Morphological Features"
)

plt.tight_layout()

plt.savefig(
    "results/correlation_heatmap.png",
    dpi=300
)

plt.close()


# ==========================================
# 7. FEATURE COMPARISON BY CLASS
# ==========================================

for feature in features:

    plt.figure(figsize=(7, 5))

    sns.boxplot(
        data=df,
        x="Class",
        y=feature
    )

    plt.title(
        f"{feature} by Rice Variety"
    )

    plt.xlabel("Rice Variety")
    plt.ylabel(feature)

    plt.tight_layout()

    filename = (
        "results/"
        + feature.lower()
        + "_by_class.png"
    )

    plt.savefig(
    filename,
    dpi=300
)

plt.close()


# ==========================================
# 8. PRINT GROUP MEANS
# ==========================================

print("\nMEAN VALUES BY RICE VARIETY - 03_eda.py:181")
print("= - 03_eda.py:182" * 60)

print(
    df.groupby("Class")[features].mean()
)