
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Download dataset
url = "https://raw.githubusercontent.com/acakin/house_price_prediction/master/train.csv"

df = pd.read_csv(url)

# Select feature and target
X = df[["GrLivArea"]]
y = df["SalePrice"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# 1. Standard Linear Regression
# --------------------------------------------------

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

r2_linear = r2_score(y_test, y_pred_linear)


# --------------------------------------------------
# 2. Polynomial Regression
# --------------------------------------------------

# Polynomial degree
degree = 2

poly = PolynomialFeatures(degree=degree)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()

poly_model.fit(X_train_poly, y_train)

y_pred_poly = poly_model.predict(X_test_poly)

r2_poly = r2_score(y_test, y_pred_poly)


# --------------------------------------------------
# Display Results
# --------------------------------------------------

print("--- R2 Score Comparison ---")

print(f"Linear Regression R2 Score:   {r2_linear:.4f}")

print(f"Polynomial Regression R2 Score: {r2_poly:.4f}")


# --------------------------------------------------
# Compare R2 Scores
# --------------------------------------------------

difference = r2_poly - r2_linear

print(f"\nDifference in R2 Score: {difference:.4f}")


# --------------------------------------------------
# Plot
# --------------------------------------------------

# Sort values for smooth polynomial curve
X_plot = pd.DataFrame({
    "GrLivArea": sorted(X["GrLivArea"])
})

X_plot_poly = poly.transform(X_plot)

y_plot_poly = poly_model.predict(X_plot_poly)

y_plot_linear = linear_model.predict(X_plot)


plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    color="blue",
    alpha=0.5,
    label="Actual Data"
)

plt.plot(
    X_plot,
    y_plot_linear,
    color="red",
    linewidth=2,
    label="Linear Regression"
)

plt.plot(
    X_plot,
    y_plot_poly,
    color="green",
    linewidth=2,
    label="Polynomial Regression"
)

plt.xlabel("House Area (sq. ft.)")
plt.ylabel("Sale Price ($)")
plt.title("Linear vs Polynomial Regression")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

plt.show()
