import numpy as np

# Bipolar activation function
def sign(x):
    return np.where(x >= 0, 1, -1)

# Training patterns (4 vectors)
patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
    [-1, -1, 1, 1]
])

# Number of neurons
n = patterns.shape[1]

# Initialize weight matrix
W = np.zeros((n, n))

# Hebbian Learning Rule
for p in patterns:
    p = p.reshape(n, 1)
    W += p @ p.T

# Remove self-connections
np.fill_diagonal(W, 0)

print("Weight Matrix:")
print(W)

# Testing with a noisy pattern
test_pattern = np.array([1, -1, 1, 1])
print("\nInitial Test Pattern:", test_pattern)

# Recall process
for _ in range(5):
    test_pattern = sign(np.dot(W, test_pattern))

print("Recovered Pattern:", test_pattern)

# Test recovery for all stored patterns
print("\n--- Testing Recovery for All Stored Patterns ---")
for i, pattern in enumerate(patterns):
    noisy = pattern.copy()
    # Add noise to first element
    noisy[0] = -noisy[0]
    print(f"\nPattern {i+1}:")
    print(f"Original: {pattern}")
    print(f"Noisy:    {noisy}")
    
    recovered = noisy.copy()
    for _ in range(5):
        recovered = sign(np.dot(W, recovered))
    print(f"Recovered: {recovered}")
