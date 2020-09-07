"""
5.4.1 곱셈 계층
"""
class Mullayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y

        return self.x * self.y

    def backward(self,dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self,x,y):
        out = x + y

        return out

    def backward(self,dout):
        dx = dout*1
        dy = dout*1

        return dx, dy

if __name__ == '__main__':
    apple = 100
    apple_num = 2
    orange = 150
    orange_num = 3
    tax = 1.1

    #계층들
    mul_apple_layer = Mullayer()
    mul_tax_layer = Mullayer()
    mul_orange_layer = Mullayer()
    add_orange_apple_layer = AddLayer()

    #순전파
    apple_price = mul_apple_layer.forward(apple,apple_num)
    orange_price = mul_orange_layer.forward(orange,orange_num)
    all_price = add_orange_apple_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price,tax)

    #역전파
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_orange_apple_layer.backward(dall_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print(price)
    print(dapple_num, dapple, dorange, dorange_num, dtax)
