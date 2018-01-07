from ch05.nn_5_4 import *


# 활성화 함수 계층 구현하기

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class sigmoid:
    def __init__(self):
        self.Y = None
        pass

    def forward(self, x):
        self.Y = 1 / (1 + np.exp(x))
        return self.Y

    def backward(self, dout):
        return dout * self.Y * (1.0 - self.Y)


