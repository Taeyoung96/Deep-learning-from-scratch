import numpy as np
from common.functions import *

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
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        #reshape를 쓰는데 이차원 배열에 행이 x.shape[0]이고 열을 알아서 맞춰준다.

        self.x = x
        out = np.dot(x,self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape) # 입력 데이터 모양 변경(텐서 대응)

        return  dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None #손실함수
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout = 1):
        batch_size = t.shape[0]

        if self.t.size == self.y.size:    #정답 레이블이 원-핫 인코딩일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

if __name__ == '__main__':
    x = np.random.rand(2)
    #print(x)
    w = np.random.rand(2,3)
    #print(w)

    dY = np.array([[1,2,3],[4,5,6]])
    dB = np.sum(dY,axis=0)

    #print(dB)

    x = [np.arange(2),[0,1,2]]
    """
    list와 배열은 다른 것!
    x는 리스트라서 배열에서 쓰는 함수를 사용할 수 없다.
    """
    print(x)


