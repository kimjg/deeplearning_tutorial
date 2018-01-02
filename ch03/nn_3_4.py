from common.functions import *

# 3층 신경망 구현하기

X = np.array([1, 2])
W1 = np.array([[1, 3, 5], [2, 4, 6]])
W2 = np.array([[1, 2], [3, 4], [5, 6]])
W3 = np.array([[1, 2], [3, 4]])

Z1 = np.dot(X, W1)
Z2 = np.dot(Z1, W2)
y = np.dot(Z2, W3)

print(y)


# 활성함수 및 편향 추가

X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
B = np.array([3, 4, 5])
y = sigmoid(np.dot(X, W) + B)

print(y)


# 출력층으로의 전달 -> identity function 사용

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

Z1 = sigmoid(np.dot(X, W1) + B1)
Z2 = sigmoid(np.dot(Z1, W2) + B2)
y = identity_function(np.dot(Z2, W3) + B3)

print(y)


# 구현정리
