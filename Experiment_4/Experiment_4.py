import numpy as np 
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score

#Dataset creation
x=np.array([1,2,3,4,5,7,8]).reshape(-1,1)
y=np.array([35,40,50,55,60,68,75])

#Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.25, random_state=42
)

# Model intialization & Training 
model=LinearRegression()
model.fit(x_train,y_train)

# Prediction
y_pred = model.predict(x_test)

# Metrics calculation

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print("Slope (b1):", model.coef_[0])
print("Intercept (b0):",model.intercept_)
print("\n---Evalution Metrics---")
print(f"MAE:{mae:.2f}")
print(f"MSE:{mse:.2f}")
print(f"RMSE:{rmse:.2f}")
print(f"R2:{r2:.4f}")

# Plotting

plt.figure(figsize=(7,5))
plt.scatter(x,y,color="blue",label="Actual Data")
plt.plot(x,model.predict(x),color="red",linewidth=2,label="Regression Line")
plt.xlabel("Study Hours")
plt.ylabel("Exam score")
plt.title("Study Hours vs Score")
plt.legend()
plt.grid(True,linestyle="--",alpha=0.6)
plt.show()