import numpy as np

# AND 게이트 구현
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.9

    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

print("\n\nAND======")
print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))


# NAND 게이트 구현
def NAND(x1, x2):
    w1, w2, theta = -0.5, -0.5, -0.9

    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

print("\n\nNAND======")
print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))

# NAND 게이트 구현
def OR(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.4

    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

print("\n\nOR======")
print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))


# numpy 로 구현 (with bias)
def AND2(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(x * w) + b

    if tmp <= 0:
        return 0
    else:
        return 1

print("\n\nAND2======")
print(AND2(0, 0))
print(AND2(0, 1))
print(AND2(1, 0))
print(AND2(1, 1))
