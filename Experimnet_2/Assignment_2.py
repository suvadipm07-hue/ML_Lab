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

# Boxplots for all numerical attributes

plt.figure(figsize=(15, 8))

sns.boxplot(data=df.drop(columns=["target"]))

plt.title("Boxplots of Wine Dataset Numerical Attributes")
plt.xlabel("Features")
plt.ylabel("Values")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()