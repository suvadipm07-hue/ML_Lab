
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Download dataset
url = "https://raw.githubusercontent.com/acakin/house_price_prediction/master/train.csv"

df = pd.read_csv(url)

# Display dataset information
print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

# Select two features
X = df[["GrLivArea", "BedroomAbvGr"]]

# Target variable
y = df["SalePrice"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model initialization and training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Display coefficients
print("\n--- Multiple Linear Regression Results ---")

print("Intercept (b0):", model.intercept_)

print("Coefficient for GrLivArea (b1):", model.coef_[0])

print("Coefficient for BedroomAbvGr (b2):", model.coef_[1])

print("\n--- Evaluation Metrics ---")
print(f"MAE: ${mae:.2f}")
print(f"MSE: ${mse:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Predict price for a new house
new_house = pd.DataFrame({
    "GrLivArea": [2000],
    "BedroomAbvGr": [3]
})

predicted_price = model.predict(new_house)

print("\n--- New House Prediction ---")
print("House Area: 2000 sq. ft.")
print("Bedrooms: 3")
print(f"Predicted Price: ${predicted_price[0]:,.2f}")

# Actual vs Predicted Plot
plt.figure(figsize=(8, 5))

plt.scatter(
    y_test,
    y_pred,
    color="blue",
    alpha=0.6
)

plt.xlabel("Actual House Price ($)")
plt.ylabel("Predicted House Price ($)")
plt.title("Actual vs Predicted House Prices")

plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
