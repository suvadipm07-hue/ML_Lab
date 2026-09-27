
import pandas as pd


from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Create Synthetic Dataset
data = {
    "Age": [22, 25, None, 30, 28, 35, None, 40],
    "Salary": [25000, 32000, 28000, None, 45000, 50000, 38000, None],
    "Department": [
        "IT", "HR", "it", "Finance",
        None, "HR", "IT", "finance"
    ],
    "Years_Experience": [1, 2, 3, None, 5, 7, 4, 10]
}

df = pd.DataFrame(data)

print("--- Original Dataset ---")
print(df)

# Correct inconsistent categorical values
df["Department"] = df["Department"].replace({
    "it": "IT",
    "finance": "Finance"
})

# Features
numeric_features = ["Age", "Salary", "Years_Experience"]
categorical_features = ["Department"]

# Numerical preprocessing using StandardScaler
standard_numeric = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Numerical preprocessing using MinMaxScaler
minmax_numeric = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

# Categorical preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# StandardScaler preprocessing
standard_preprocessor = ColumnTransformer(
    transformers=[
        ("num", standard_numeric, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# MinMaxScaler preprocessing
minmax_preprocessor = ColumnTransformer(
    transformers=[
        ("num", minmax_numeric, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Apply StandardScaler
X_standard = standard_preprocessor.fit_transform(df)

# Apply MinMaxScaler
X_minmax = minmax_preprocessor.fit_transform(df)

# Display StandardScaler results
print("\n--- StandardScaler Transformed Array ---")
print(X_standard)

print("\n--- StandardScaler Numerical Range ---")
print("Minimum:", X_standard[:, :3].min())
print("Maximum:", X_standard[:, :3].max())

# Display MinMaxScaler results
print("\n--- MinMaxScaler Transformed Array ---")
print(X_minmax)

print("\n--- MinMaxScaler Numerical Range ---")
print("Minimum:", X_minmax[:, :3].min())
print("Maximum:", X_minmax[:, :3].max())

