import numpy as np
from ch02.perceptron_2_3 import *

# 다층 perceptron
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print("\n\nXOR======")
print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))

# NAND 게이트로 컴퓨터를 구현할 수 있다
# 따라서, 이론상 퍼셉트론 층을 거듭 쌓으면 비선형적인 표현도 가능하고 이론상 컴퓨터가 수행하는 처리도 모두 표현할 수 있다.

