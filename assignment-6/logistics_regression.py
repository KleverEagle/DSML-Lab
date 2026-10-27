import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# 1. Load the CSV Data
data = np.genfromtxt('logistics_data.csv', delimiter=',', skip_header=1)
X = data[:, :2]  # First two columns are features
y = data[:, 2:]  # Last column is the binary target (0 or 1)


class LogisticRegression:
    def __init__(self):
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: npt.NDArray, y: npt.NDArray, epochs: int = 5000, learning_rate: float = 0.01):
        m = len(y)

        if self.weights is None:
            self.weights = np.zeros((X.shape[1], 1))
            self.bias = 0.0

        for epoch in range(epochs):
            # Forward Pass
            z = X @ self.weights + self.bias
            probabilities = self.sigmoid(z)

            # Error
            error_lin = probabilities - y

            # Gradient
            # Same as that of linear regression
            # Mix of sigmoid and log loss function cancels the effect
            # Log loss = -(1/m) * (y.log(p(y)) + (1-y)log(1-p(y)))
            gradient_weights = (X.T @ error_lin) / m
            gradient_bias = np.sum(error_lin) / m

            self.weights -= learning_rate * gradient_weights
            self.bias -= learning_rate * gradient_bias

    def predict(self, x):
        probabilities = self.sigmoid(x @ self.weights + self.bias)
        return (probabilities >= 0.5).astype(int)


model = LogisticRegression()
model.fit(X, y)

plt.figure(figsize=(8, 6))

plt.scatter(X[y[:, 0] == 0][:, 0], X[y[:, 0] == 0][:, 1], color='red', label='Class 0', marker='o')

plt.scatter(X[y[:, 0] == 1][:, 0], X[y[:, 0] == 1][:, 1], color='blue', label='Class 1', marker='s')

# 4. Calculate and plot the Decision Boundary
# The boundary occurs where z = 0 (which means sigmoid(0) = 0.5)
# Equation: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
w1 = model.weights[0, 0]
w2 = model.weights[1, 0]
b = model.bias

# Create two x1 points spanning the width of our plot to draw the line
x1_line = np.array([np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1])
x2_line = -(w1 * x1_line + b) / w2

plt.plot(x1_line, x2_line, color='green', linestyle='--', label='Decision Boundary')

# 5. Plot aesthetics
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Prevent the line from stretching the axes too far
plt.xlim(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1)
plt.ylim(np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1)

plt.show()