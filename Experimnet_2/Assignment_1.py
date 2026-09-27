import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import load_wine

#Load dataset
wine =load_wine()
df =pd.DataFrame(wine.data,columns=wine.feature_names)
df["target"]= wine.target

#Basic Data exploration
print("--- First Five Rows ---")
print(df.head())

print("---Dataset Information ---")
print(df.info())

print("--- Statistical Summary ---")
print(df.describe())

print("--- Missing values ---")
print(df.isnull().sum())

# Correlation matrix

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

plt.title("Alcohol vs Malic Acid")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")

plt.show()