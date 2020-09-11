import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.util import im2col , col2im #image to column

"""
합성곱 계층 구현하기
"""
x1 = np.random.rand(1,3,7,7) #데이터의 수, 채널 수, 높이, 너비
col1 = im2col(x1,5,5,stride=1,pad=0)
print(col1.shape) #(stride가 움직인 횟수, 채널 수*높이*너비)




class Convolution:
    def __init__(self, W, b, stride = 1, pad = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x,FH,FW,self.stride,self.pad)
        col_W = self.W.reshape(FN,-1).T  #전치 행렬 사용 ,-1은 FN 나머지 원소들을 알아서 묶어주는 역할

        out = np.dot(col,col_W) + self.b
        out = out.reshape(N,out_h, out_w, -1).transpose(0,3,1,2) #shape을 적혀진 우선 순위 대로 바꾼다.

        self.x =x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout) #전치행렬을 이용하여 곱이 될 수 있도록 형태를 맞춰준다.
        self.dW = self.dW.transpose(1,0).reshape(FN,C,FH,FW) #2차원 형태를 다시 4차원으로 바꿔준다.

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx

"""
풀링 계층 구현하기
1. 입력 데이터 전개
2. 행별 최댓값 구한다.
3. 적절한 모양으로 변형
"""

class Pooling:
    def __init__(self, pool_h, pool_w, stride = 1, pad = 0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        #전개
        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        #최댓값 구하기
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)

        #변형
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.reshape(0,2,3,1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size,pool_size))
        dmax[np.arange(self.arg_max.size),self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


