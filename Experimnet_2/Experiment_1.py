import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.datasets import load_iris

#Load dataset
iris =load_iris()
df =pd.DataFrame(iris.data,columns=iris.feature_names)
df["target"]= iris.target

#Basic Data exploration
print("--- First Five Rows ---")
print(df.head())

print("---Dataset Information ---")
print(df.info())

print("--- Statistical Summary ---")
print(df.describe())

print("--- Missing values ---")
print(df.isnull().sum())

#Correlation matrix

print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df,
    x="sepal length (cm)",
    y="petal length (cm)",
    hue="target",
    palette="viridis"
)

plt.title("Sepal Length vs Petal Length")
plt.show()

