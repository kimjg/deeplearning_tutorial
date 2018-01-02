import numpy as np
import matplotlib.pyplot as plt


# 수치 미분

# 나쁜 예
def numerical_diff_inefficent(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h

# 오차 존재
def numerical_diff_inefficent2(f, x):
    h = 1e-4
    return (f(x + h) - f(x)) / h

# 중심 차분 or 중앙 차분
def numerial_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


# 수치 미분

def f_1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0, 20, 0.1)
y = f_1(x)
plt.plot(x, y, linestyle="--",label="f(x)")


y_ = numerial_diff(f_1, x)
plt.plot(x, y_, label="f'(x)")
plt.show()


# 4.3.3 편미분

def f_2(x):
    return x[0]**2 + x[1]**2


def numerical_gradients(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp_val = x[i]
        x[i] = tmp_val + h
        # 함수가 선형 조건을 만족한다면, x 변수 1개로만 미분해도 된다.
        fxh1 = f(x)
        x[i] = tmp_val - h
        fxh2 = f(x)
        grad[i] = (fxh1 - fxh2) / (2 * h)

        x[i] = tmp_val

    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        x -= lr * numerical_gradients(f, x)

    return x


