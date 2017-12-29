import numpy as np

# array 조건 넣어 값 추출하기
a = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
b = np.array([False, True, True, True, True, True, True, True, False, False])
print(a > 3)
print(a[b])
print(a[a > 3])

# 조건에 맞는 원소 % 출력
print(np.mean(b))

# 다차원 배열 1차원으로 reshape
c = np.array([[1, 2], [3, 4]])
print(c.flatten())
print(c.reshape([c.shape[0] * c.shape[1]]))