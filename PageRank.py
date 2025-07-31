import numpy as np

#Step 1: Input edges and build adjacency matrix.
edges = [
    (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
    (2, 1), (2, 3), (2, 4), (2, 5), (2, 7),
    (3, 1), (3, 2), (3, 4), (3, 5), (3, 6),
    (4, 1), (4, 2), (4, 3),
    (5, 1), (5, 2), (5, 3), (5, 6),
    (6, 1), (6, 3), (6, 4), (6, 9),
    (7, 2), (7, 8), (7, 9),
    (8, 5), (8, 7),
    (9, 6), (9, 7)
]

N = 9  #Total number of nodes
d = 0.85  #Damping factor
tol = 1e-6  #Convergence tolerance
max_iter = 100

#Step 2:Build adjacency matrix(column-normalized).
A = np.zeros((N, N))

for from_node, to_node in edges:
    A[to_node - 1][from_node - 1] = 1  #Note:column = from_node, row = to_node

#Normalize columns(handling sink nodes)
for j in range(N):
    col_sum = np.sum(A[:, j])
    if col_sum != 0:
        A[:, j] /= col_sum
    else:
        A[:, j] = 1.0 / N  #Distribute uniformly for sink node

#Step 3:Power iteration to compute PageRank.
R = np.ones(N) / N  #Initial rank
for _ in range(max_iter):
    new_R = (1 - d) / N + d * A @ R
    if np.linalg.norm(new_R - R, ord=1) < tol:
        break
    R = new_R

#Step 4:Output highest ranked node
highest_node = np.argmax(R) + 1  #Add 1 to match 1-indexed nodes.
print(f"Highest ranked node: {highest_node}")
print(f"PageRank values: {np.round(R, 4)}")
