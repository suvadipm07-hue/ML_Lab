
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

# Select only house area and price
X = df[["GrLivArea"]]
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

print("\n--- Linear Regression Results ---")
print("Slope (b1):", model.coef_[0])
print("Intercept (b0):", model.intercept_)

print("\n--- Evaluation Metrics ---")
print(f"MAE: ${mae:.2f}")
print(f"MSE: ${mse:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# Predict price for a new house
area = 2000

predicted_price = model.predict([[area]])

print(f"\nPredicted price for {area} sq. ft. house: "
      f"${predicted_price[0]:,.2f}")

# Plot actual data and regression line
plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    color="blue",
    label="Actual Prices"
)

plt.plot(
    X_test,
    y_pred,
    color="red",
    linewidth=2,
    label="Regression Line"
)

plt.xlabel("House Area (sq. ft.)")
plt.ylabel("Sale Price ($)")
plt.title("House Area vs Sale Price")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()

