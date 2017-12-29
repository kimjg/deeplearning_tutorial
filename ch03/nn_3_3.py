import numpy as np

# 행렬의 내적

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))


# 3.3.3 신경망의 내적

X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
y = np.dot(X, W)

print(y)

