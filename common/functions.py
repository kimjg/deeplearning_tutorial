import numpy as np


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x

# 배치용 손실함수
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7

    return -1 * np.sum(t * np.log(y + delta)) / batch_size

# exp 값이 매우 커지면 overflow 문제가 발생한다. e^1000 = inf 가 나온다.
# 개선한 소프트맥수 함수

def softmax(x):

    max_val = np.max(x)
    x = x - max_val
    return np.exp(x) / np.sum(np.exp(x))

