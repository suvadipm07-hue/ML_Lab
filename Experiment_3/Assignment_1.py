```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

# Deliberately create inconsistent categorical values
df["Department"] = df["Department"].replace({
    "it": "IT",
    "finance": "Finance"
})

print("\n--- After Correcting Inconsistent Values ---")
print(df)

# Separate Numerical and Categorical Features
numeric_features = ["Age", "Salary", "Years_Experience"]
categorical_features = ["Department"]

# Numerical Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical Preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Apply Preprocessing
X_processed = preprocessor.fit_transform(df)

# Display Results
print("\n--- Processed Feature Matrix ---")
print(X_processed.toarray())

print("\n--- Processed Feature Matrix Shape ---")
print(X_processed.shape)
```
