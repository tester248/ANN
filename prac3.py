import numpy as np

def ascii_binary(digit):
    """Return 8-bit ASCII binary representation (as float array) of the digit character."""
    ascii_val = ord(str(digit))
    binary = format(ascii_val, '08b')
    return np.array([int(bit) for bit in binary], dtype=np.float64)


def train_perceptron(X, y, learning_rate=0.1, epochs=50):
    weights = np.zeros(X.shape[1], dtype=np.float64)
    bias = 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            net_input = np.dot(xi, weights) + bias
            output = 1 if net_input >= 0 else 0
            error = yi - output
            weights += learning_rate * error * xi
            bias += learning_rate * error
    return weights, bias


def predict(x, weights, bias):
    return 1 if (np.dot(x, weights) + bias) >= 0 else 0


def main():
    # Prepare dataset: ASCII binary of characters '0'..'9'
    X = np.array([ascii_binary(i) for i in range(10)])
    y = np.array([1 if i % 2 == 0 else 0 for i in range(10)], dtype=np.int32)

    weights, bias = train_perceptron(X, y, learning_rate=0.1, epochs=50)

    print("Testing Perceptron for Even/Odd Recognition:\n")
    for i in range(10):
        test_input = ascii_binary(i)
        prediction = predict(test_input, weights, bias)
        print(f"Digit {i} -> {'Even' if prediction == 1 else 'Odd'}")

    # Report training accuracy and parameters
    preds = np.array([predict(x, weights, bias) for x in X])
    acc = np.mean(preds == y)
    print(f"\nTraining accuracy: {acc * 100:.1f}%")
    print("Weights:", weights)
    print("Bias:", bias)


if __name__ == "__main__":
    main()
