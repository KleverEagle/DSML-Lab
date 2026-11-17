import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

data = np.genfromtxt('linear_regression.csv', delimiter=',', skip_header=1)
X = data[:, :2]
y = data[:, 2:]

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    # y is a 2D array of [m,1] not 1D
    def fit(self, X: npt.NDArray, y: npt.NDArray, epochs: int = 5000, learning_rate: float = 0.01):
        m = len(y)

        if self.weights is None:
            self.weights = np.zeros((X.shape[1], 1))
            self.bias = 0.0

        for epoch in range(epochs):
            prediction = self.predict(X)

            error = prediction - y

            # Gradient
            gradient_weights = (X.T @ error) / m
            gradient_bias = np.sum(error) / m

            # 5. Update weights and bias
            self.weights -= learning_rate * gradient_weights
            self.bias -= learning_rate * gradient_bias

    def predict(self, x):
        return x @ self.weights + self.bias

model = LinearRegression()
model.fit(X, y)
print(model.weights)
print(model.bias)

predictions = model.predict(X)

plt.figure(figsize=(8, 6))

plt.scatter(y, predictions, color='blue', alpha=0.7, label='Model Predictions')

min_val = min(np.min(y), np.min(predictions))
max_val = max(np.max(y), np.max(predictions))
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Fit (y = x)')

plt.title("Actual vs. Predicted Targets")
plt.xlabel("Actual Target Values (y)")
plt.ylabel("Predicted Values")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.show()
