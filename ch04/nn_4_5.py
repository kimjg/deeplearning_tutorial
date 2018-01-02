import numpy as np
from common.functions import *
from common.gradient import *

# 다층 신경망 구현


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.rand(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.rand(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):

        A1 = np.dot(x, self.params['W1']) + self.params['b1']
        Z1 = sigmoid(A1)
        Z2 = np.dot(Z1, self.params['W2']) + self.params['b2']
        y = softmax(Z2)

        return y

    def loss(self, x, t):

        y = self.predict(x)

        return cross_entropy_error(t, y)

    def accuracy(self, x, t):
        y = self.predict(x)

        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        return np.mean(y == t)

    def numerical_gradient(self, x, t):
        f = lambda w: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(f, self.params['W1'])
        grads['b1'] = numerical_gradient(f, self.params['b1'])
        grads['W2'] = numerical_gradient(f, self.params['Ww'])
        grads['b2'] = numerical_gradient(f, self.params['b1'])

        return grads

net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
print(net.params['W1'].shape)
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b1'].shape)

x = np.random.rand(100, 784)
t = np.random.rand(100, 10)
y = net.predict(x)

grads = net.numerical_gradient(x, t)

print(grads.shape)

