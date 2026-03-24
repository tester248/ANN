import numpy as np
import matplotlib.pyplot as plt


def train_perceptron(X, y, learning_rate=0.1, epochs=10):
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            net_input = np.dot(xi, w) + b
            y_pred = 1 if net_input >= 0 else 0
            error = yi - y_pred
            w += learning_rate * error * xi
            b += learning_rate * error
    return w, b


def plot_decision_regions(X, y, w, b, filename="prac5_decision_regions.png"):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], w) + b
    Z = (Z >= 0).astype(int).reshape(xx.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

    mask0 = (y == 0)
    mask1 = (y == 1)
    plt.scatter(X[mask0, 0], X[mask0, 1], marker='o', s=100, edgecolors='k', color='C0', label='0 (output)')
    plt.scatter(X[mask1, 0], X[mask1, 1], marker='x', s=100, color='C3', label='1 (output)')

    # decision boundary line (if w[1] != 0)
    if abs(w[1]) > 1e-8:
        x_vals = np.array([x_min, x_max])
        y_vals = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals, 'k--', linewidth=1)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.title("Perceptron Decision Regions (OR Gate)")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved decision region plot to {filename}")


def main():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float64)

    # OR gate targets
    y = np.array([0, 1, 1, 1], dtype=np.int32)

    w, b = train_perceptron(X, y, learning_rate=0.1, epochs=10)

    print("Final Weights:", w)
    print("Final Bias:", b)

    plot_decision_regions(X, y, w, b)


if __name__ == "__main__":
    main()
