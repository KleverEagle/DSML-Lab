import math
import random


class MultiLayerPerceptron:
    def __init__(self, learning_rate=0.5):
        random.seed(42)
        self.W_hidden = [
            [random.uniform(-1, 1), random.uniform(-1, 1)],  # Weights from x1 to [h1, h2]
            [random.uniform(-1, 1), random.uniform(-1, 1)]  # Weights from x2 to [h1, h2]
        ]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]  # Biases for [h1, h2]

        self.W_output = [random.uniform(-1, 1), random.uniform(-1, 1)]  # Weights from [h1, h2] to output
        self.b_output = random.uniform(-1, 1)  # Bias for output

        self.lr = learning_rate

    @staticmethod
    def sigmoid(z):
        # z = max(min(z, 250), -250)
        return 1.0 / (1.0 + math.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        # Derivative of sigmoid requires the activated output 'a'
        return a * (1.0 - a)

    def forward(self, x1, x2):
        # Hidden Layer
        z1 = x1 * self.W_hidden[0][0] + x2 * self.W_hidden[1][0] + self.b_hidden[0]
        h1 = self.sigmoid(z1)

        z2 = x1 * self.W_hidden[0][1] + x2 * self.W_hidden[1][1] + self.b_hidden[1]
        h2 = self.sigmoid(z2)

        # Output Layer
        z_out = h1 * self.W_output[0] + h2 * self.W_output[1] + self.b_output
        out = self.sigmoid(z_out)

        return (h1, h2), out

    def train(self, training_data, epochs):
        for epoch in range(epochs):
            total_loss = 0

            for (x1, x2), target in training_data:
                # --- Forward Pass ---
                (h1, h2), out = self.forward(x1, x2)

                # Calculate Mean Squared Error Loss
                loss = (target - out) ** 2
                total_loss += loss

                # Backpropagation

                # Derivative of (y - yt)^2 = 2 (y - yt)
                error_derivative = 2 * (out - target)

                delta_out = error_derivative * self.sigmoid_derivative(out)

                # Calculate the error delta at the hidden layer
                delta_h1 = (delta_out * self.W_output[0]) * self.sigmoid_derivative(h1)
                delta_h2 = (delta_out * self.W_output[1]) * self.sigmoid_derivative(h2)

                # Gradient Descent Update

                self.W_output[0] -= self.lr * delta_out * h1
                self.W_output[1] -= self.lr * delta_out * h2
                self.b_output -= self.lr * delta_out

                self.W_hidden[0][0] -= self.lr * delta_h1 * x1
                self.W_hidden[1][0] -= self.lr * delta_h1 * x2
                self.b_hidden[0] -= self.lr * delta_h1

                self.W_hidden[0][1] -= self.lr * delta_h2 * x1
                self.W_hidden[1][1] -= self.lr * delta_h2 * x2
                self.b_hidden[1] -= self.lr * delta_h2

    def predict(self, x1, x2):
        _, raw_output = self.forward(x1, x2)
        prediction = 1 if raw_output > 0.5 else 0
        return raw_output, prediction


xor_data = [
    ((0, 0), 0),
    ((0, 1), 1),
    ((1, 0), 1),
    ((1, 1), 0)
]

mlp = MultiLayerPerceptron(learning_rate=0.5)

mlp.train(xor_data, epochs=400)

# Weights Found
print(f'Weights from x1 to [h1, h2]: {mlp.W_hidden[0]}')
print(f'Weights from x2 to [h1, h2]: {mlp.W_hidden[1]}')

print(f'Biases for [h1, h2]: {mlp.b_hidden}')

print(f'Weights from [h1,h2] to out: {mlp.W_output}')
print(f'Bias for out: {mlp.b_output}')

print("XOR Gate Output:")
for (x1, x2), _ in xor_data:
    raw_out, binary_out = mlp.predict(x1, x2)
    print(f"[{x1}, {x2}] -> Raw: {raw_out:.5f} -> Binary: {binary_out}")