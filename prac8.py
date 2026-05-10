import numpy as np

# Input patterns (binary)
X = np.array([
    [1, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 1, 0]
])

# Parameters
num_features = X.shape[1]
vigilance = 0.6
weights = [np.ones(num_features,dtype=float)]

def match_score(input_vec, weight_vec):
    s = np.sum(input_vec)
    return 0.0 if s == 0 else np.sum(np.minimum(input_vec, weight_vec)) / s


for input_vec in X:
    # compute scores and sort categories by score desc
    scores = [match_score(input_vec, w) for w in weights]
    tried = set()
    assigned = False
    while len(tried) < len(weights):
        j = int(np.argmax([s if idx not in tried else -1 for idx, s in enumerate(scores)]))
        tried.add(j)
        if scores[j] >= vigilance:
            # accept and update prototype
            weights[j] = np.minimum(weights[j], input_vec)
            assigned = True
            break
        # else continue to next best
    if not assigned:
        # create new cluster with weight = input prototype
        weights.append(input_vec.copy())

# Testing
print("Final cluster weights:")
print(weights)
print(f"Number of clusters formed: {len(weights)}")
print("\nCluster assignment:")
for input_vec in X:
    scores = [match_score(input_vec, w) for w in weights]
    print(f"Input: {input_vec} -> Cluster: {np.argmax(scores)}")
