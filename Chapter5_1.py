import numpy as np

"""
ReLU 클래스 구현
"""

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

"""
Sigmoid 함수 구현
"""
class sigmoid:
    def __init__(self):
        self.out = None

    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx

"""
Affine 함수 구현
"""
class Affine:
    def __init__(self,W,b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None

        #가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self,x):
        #텐서 대응



if __name__ == '__main__':
    x = np.random.rand(2)
    print(x)
    w = np.random.rand(2,3)
    print(w)

    dY = np.array([[1,2,3],[4,5,6]])
    dB = np.sum(dY,axis=0)

    print(dB)


