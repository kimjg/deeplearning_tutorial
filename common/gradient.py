import numpy as np


# def numerical_gradients(f, x):
#     h = 1e-4
#     grad = np.zeros_like(x)
#
#     for i in range(x.size):
#         tmp_val = x[i]
#         x[i] = tmp_val + h
#         # 함수가 선형 조건을 만족한다면, x 변수 1개로만 미분해도 된다.
#         fxh1 = f(x)
#         x[i] = tmp_val - h
#         fxh2 = f(x)
#         grad[i] = (fxh1 - fxh2) / (2 * h)
#
#         x[i] = tmp_val
#
#     return grad
#

def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        x -= lr * numerical_gradient(f, x)

    return x

