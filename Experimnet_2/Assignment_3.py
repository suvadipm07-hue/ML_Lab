import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine

# Load dataset
wine = load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)

df["target"] = wine.target

# Basic Data Exploration

print("--- First Five Rows ---")
print(df.head())

print("--- Dataset Information ---")
print(df.info())

print("--- Statistical Summary ---")
print(df.describe())

print("--- Missing Values ---")
print(df.isnull().sum())

# Correlation Matrix

print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))

# Scatter Plot

plt.figure(figsize=(7, 5))

sns.scatterplot(
    data=df,
    x="alcohol",
    y="malic_acid",
    hue="target",
    palette="viridis"
)
# Correlation Heatmap

plt.figure(figsize=(12, 8))

corr_matrix = df.drop(columns=["target"]).corr()

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap - Wine Dataset")
plt.tight_layout()
plt.show()


# Find strongest positive correlation

corr_pairs = corr_matrix.unstack()

# Remove self-correlations (1.00)
corr_pairs = corr_pairs[corr_pairs < 1]

# Find strongest positive correlation
strongest_pair = corr_pairs.idxmax()
strongest_value = corr_pairs.max()

print("\n--- Strongest Positive Correlation ---")
print("Feature 1:", strongest_pair[0])
print("Feature 2:", strongest_pair[1])
print("Correlation:", strongest_value)