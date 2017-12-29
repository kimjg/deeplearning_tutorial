import numpy as np

# 출력층 설계


# 항등 함수와 소프트맥스 함수 구현하기


def softmax_ineffient(x):
    return np.exp(x) / np.sum(np.exp(x))

a = np.array([1, 2, 3])
b = np.array([1010, 1000, 990])

print(softmax_ineffient(a))
# print(softmax_ineffient(b)) -> overflow error

# exp 값이 매우 커지면 overflow 문제가 발생한다. e^1000 = inf 가 나온다.
# 개선한 소프트맥수 함수

def softmax(x):

    max_val = np.max(x)
    x = x - max_val
    return np.exp(x) / np.sum(np.exp(x))

print(softmax(b))

# 소프트맥스 함수의 출력 원소들의 합은 1이므로 확률적으로 사용한다.

# 확률이 필요하지 않을 때에는 소프트맥스를 생략하면 좋다. (성능)

