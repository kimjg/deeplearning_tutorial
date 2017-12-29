import numpy as np

# 손실 함수


print(np.random.choice(100, 10))
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

def cross_entropy_error(y, t):
    delta = 1e-7 # log0 무한대이기 때문에 아주 작은 값을 더 해준다
    return -1 * np.sum(t * np.log(y + delta))


t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(mean_squared_error(y, t))
print(cross_entropy_error(y, t))
y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.6, 0.1, 0.0, 0.0])
print(mean_squared_error(y, t))
print(cross_entropy_error(y, t))


# 배치용 손실함수
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    delta = 1e-7

    return -1 * np.sum(t * np.log(y + delta)) / batch_size

