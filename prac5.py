import numpy as np

# Activation function (sign function)
def sign(x):
    return np.where(x >= 0, 1, -1)

# Training BAM
def train_bam(X, Y):
    W = np.zeros((X.shape[1], Y.shape[1]))
    for i in range(len(X)):
        W += np.outer(X[i], Y[i])
    return W

# Recall from X to Y
def recall_Y(X, W):
    return sign(np.dot(X, W))

# Recall from Y to X
def recall_X(Y, W):
    return sign(np.dot(Y, W.T))

# Example patterns (bipolar: -1, 1)
X = np.array([
    [1, -1, 1],
    [-1, 1, -1]
])

Y = np.array([
    [1, 1],
    [-1, -1]
])

# Train BAM
W = train_bam(X, Y)
print("Weight Matrix:\n", W)

# Test recall
test_X = np.array([1, -1, 1])
predicted_Y = recall_Y(test_X, W)
print("Recalled Y:", predicted_Y)

# Reverse recall
test_Y = np.array([1, 1])
predicted_X = recall_X(test_Y, W)
print("Recalled X:", predicted_X)
