import numpy as np
import matplotlib.pyplot as plt
# 활성함수

# 계단함수


def step_function_int(x):
    if x > 0:
        return 1
    else:
        return 0


def step_function(x):
    y = x > 0
    return y.astype(np.int)

x = np.arange(-1, 1, 0.01)
y1 = step_function(x)
#
# plt.plot(x, y)
# plt.legend()
# plt.show()


# 시그모이드
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# x = np.arange(-5, 5, 0.1)
y2 = sigmoid(x)


# ReLU
def relu(x):
    return np.maximum(0, x)

# x = np.arange(-5, 5, 0.1)
y3 = relu(x)


plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.legend()
plt.show()
