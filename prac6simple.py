import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  

def sigmoid_derivative(a):
    return a * (1 - a)

def train_nn(X, y, hidden_size=4, epochs=5000, lr=0.5):
    np.random.seed(42)
    W1 = np.random.randn(X.shape[1], hidden_size) 
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, 1)
    b2 = np.zeros((1, 1))
    
    for epoch in range(epochs):
        # Forward
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = sigmoid(z2)
        
        # Loss
        loss = np.mean((y - a2) ** 2)
        
        # Backward
        m = X.shape[0]
        dz2 = a2 - y
        dW2 = np.dot(a1.T, dz2) / m
        db2 = np.mean(dz2, axis=0, keepdims=True)
        dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.mean(dz1, axis=0, keepdims=True)
        
        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return W1, b1, W2, b2

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Train
W1, b1, W2, b2 = train_nn(X, y, epochs=5000, lr=0.5)

# Test
print("\nPredictions:")
z1 = np.dot(X, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)
for i in range(len(X)):
    print(f"Input: {X[i]} → Predicted: {a2[i][0]:.4f}, Actual: {y[i][0]}")