import numpy as np


# 단점 : 한 축의 방향이 일방적일 경우, 다른쪽은 학습되지 않은 상태로 학습이 종료된다.
class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):

        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] -= self.v[key]


class AdaGrad:
    def __init__(self, lr):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * (1 / (np.sqrt(self.h[key]) + 1e-7)) * grads[key]


# class Adam:
#     def __init__(self):
#         self.lr = lr
#         self.h = None