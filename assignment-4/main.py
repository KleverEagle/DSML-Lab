import math

class MultiLayerPerceptron:
    def __init__(self):
        # h1 acts as OR gate, h2 acts as NAND gate
        self.W_hidden = [
            [6.0, 6.0],   # Weights for h1 (OR)
            [-6.0, -6.0]  # Weights for h2 (NAND)
        ]
        self.b_hidden = [-3.0, 9.0]  # Biases for [h1, h2]

        self.W_output = [6.0, 6.0]   # Output weights (AND gate)
        self.b_output = -9.0         # Output bias

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + math.exp(-z))

    def forward(self, x1, x2):
        # Hidden Layer
        z1 = x1 * self.W_hidden[0][0] + x2 * self.W_hidden[0][1] + self.b_hidden[0]
        h1 = self.sigmoid(z1)

        z2 = x1 * self.W_hidden[1][0] + x2 * self.W_hidden[1][1] + self.b_hidden[1]
        h2 = self.sigmoid(z2)

        # Output Layer
        z_out = h1 * self.W_output[0] + h2 * self.W_output[1] + self.b_output
        raw_output = self.sigmoid(z_out)

        prediction = 1 if raw_output > 0.5 else 0
        return raw_output, prediction


# Execution Test
mlp_sigmoid = MultiLayerPerceptron()

print("Corrected Sigmoid Outputs:")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    raw_out, binary_out = mlp_sigmoid.forward(x1, x2)
    print(f"[{x1}, {x2}] -> Raw: {raw_out:.5f} -> Binary: {binary_out}")