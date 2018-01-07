import numpy as np
from common.functions import *
from common.gradient import *
import matplotlib.pyplot as plt

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
        grads['W2'] = numerical_gradient(f, self.params['W2'])
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

# grads = net.numerical_gradient(x, t)

# 미니배치 학습 구현
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

for i in range(iters_num):

    if i % 10:
        print('i is ', i)

    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)

    train_loss_list.append(loss)

x_ = np.arange(0, iters_num, 1)
y_ = train_loss_list

plt.plot(x_, y_)
plt.show()
