import numpy as np

# Input patterns (binary)
X = np.array([
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 0]
])

# Parameters
num_features = X.shape[1]
num_clusters = 2
vigilance = 0.6

# Initialize weights
weights = np.ones((num_clusters, num_features))

def match_score(input_vec, weight_vec):
    return np.sum(np.minimum(input_vec, weight_vec)) / np.sum(input_vec)

# Training ART1
for input_vec in X:
    assigned = False
    for j in range(num_clusters):
        score = match_score(input_vec, weights[j])
        if score >= vigilance:
            weights[j] = np.minimum(weights[j], input_vec)
            assigned = True
            break
    if not assigned:
        print("New cluster needed (increase cluster count)")

# Testing
print("Final cluster weights:")
print(weights)
print("\nCluster assignment:")
for input_vec in X:
    scores = [match_score(input_vec, w) for w in weights]
    print(f"Input: {input_vec} -> Cluster: {np.argmax(scores)}")
