import numpy as np


class MulLayer:
    def __init__(self):
        self.X = None
        self.Y = None

    def forward(self, x, y):
        self.X = x
        self.Y = y
        return x * y

    def backward(self, dout):
        dx = dout * self.Y
        dy = dout * self.X
        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout
        dy = dout
        return dx, dy

if __name__ == "__main__":
    # 사과 2개

    apple = 100
    apple_num = 2
    tax = 1.1

    mul_sumlayer = MulLayer()
    mul_taxlayer = MulLayer()

    z1 = mul_sumlayer.forward(apple, apple_num)
    z2 = mul_taxlayer.forward(z1, tax)

    # print(z2)

    dprice = 1
    dapple_price, dtax = mul_taxlayer.backward(dprice)
    dapple, dapple_num = mul_sumlayer.backward(dapple_price)

    # print(dapple, dapple_num, dtax)



    # 사과 2개, 귤 3개
    apple = 100
    apple_num = 2
    tangerine = 150
    tangerine_num = 3
    tax = 1.1

    mul_appleLayer = MulLayer()
    mul_tangerineLayer = MulLayer()
    add_fruitsLayer = AddLayer()
    mul_taxLayer = MulLayer()

    apple_price = mul_appleLayer.forward(apple, apple_num)
    tangerine_price = mul_tangerineLayer.forward(tangerine, tangerine_num)
    fruits_price = add_fruitsLayer.forward(apple_price, tangerine_price)
    price = mul_taxLayer.forward(fruits_price, tax)

    print(price)

    dprice = 1

    dfruits_price, dtax = mul_taxLayer.backward(dprice)
    dapple_price, dtangerine_price = add_fruitsLayer.backward(dfruits_price)
    dapple, dapple_num = mul_appleLayer.backward(dapple_price)
    dtangerine, dtangerine_num = mul_tangerineLayer.backward(dtangerine_price)

    print(dapple_num, dapple, dtangerine, dtangerine_num, dtax)