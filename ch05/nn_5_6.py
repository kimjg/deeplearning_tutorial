
import numpy as np
from common.functions import *

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.X = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dX = np.dot(dout, np.transpose(self.W))
        dW = np.dot(np.transpose(self.X), dout)
        db = np.sum(dout, axis=0)
        self.dW = dW
        self.db = db
        return dX


class Softmax:

    def __init__(self):
        self.t = None
        self.y = None
        pass

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        loss = cross_entropy_error(y=self.y, t=self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

