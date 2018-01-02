from common.functions import *
from common.gradient import *

# 단층 신경망 구현


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return cross_entropy_error(y, t)


net = simpleNet()
print(net.W)
x = np.array([[0.9, 0.1], [0.6, 0.9]])
print(np.argmax(net.predict(x), axis=1))

t = np.array([[1, 0, 0], [0, 0, 1]]) # 0, 2

f = lambda w: net.loss(x, t)

#
# grad = numerical_gradient(loss, net.W)
# print(grad)

# 학습
epoch = 10
for i in range(epoch):
    gradient_descent(f, net.W)

p = np.argmax(net.predict(x), axis=1)
print(net.W)
print(p)

