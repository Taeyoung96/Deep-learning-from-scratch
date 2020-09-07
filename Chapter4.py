import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import sys,os
sys.path.append(os.pardir)
from mnist import load_mnist
"""
4.2.1
평균 제곱 오차 구하기 (MSE)
"""
def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

"""
4.2.2
교차 엔트로피 오차 구하기 (Cross Entropy Error) - minibatch용
"""
def cross_entropy_error(y,t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)

    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

"""
4.2.3
미니배치 학습
"""
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

"""
4.3.1 미분 - 반올림 오차를 고려하여 미분 함수를 만들어 준다.
"""

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return x[0]**2 + x[1]**2

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)

plt.xlabel("x")
plt.ylabel("f(x)")

plt.plot(x,y)
#plt.show()

print(numerical_diff(function_1,5))

"""
4.4
기울기 계산
"""
def numerical_diff_array(f,x):        #배열 형태로 기울기 구하기
    h = 1e-4
    grad = np.zeros_like(x)    #x와 크기가 같은 배열 하나 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        #f(x+h) 계산
        x[idx] = tmp_val + h
        fx1 = f(x)
        #f(x-h) 계산
        x[idx] = tmp_val - h
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2*h)
        x[idx] = tmp_val  #값 복원을 해줘야 한다.

    return grad

def numerical_gradient(f,x):
    if x.ndim == 1:
        return numerical_diff_array(f,x)
    else:
        grad = np.zeros_like(x)

        for idx, val in enumerate(x):  #enumerate는 반복문 형태로 인덱스 원소와 안의 원소를 튜플로 반환
            grad[idx] = numerical_diff_array(f,val)

        return grad


def function(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

def targent_line(f,x):
    d = numerical_gradient(f,x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

x0 = np.arange(-2, 2.5, 0.25)
x1 = np.arange(-2, 2.5, 0.25)
X, Y = np.meshgrid(x0, x1)  #행단위와 열단위로 각각 해당 배열의 정방 행렬을 선언

X = X.flatten()  #1차원 배열로 만들어준다
Y = Y.flatten()

grad = numerical_gradient(function, np.array([X, Y]) )

plt.figure()      #새로운 figure 생성
plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666") # 벡터 field 그리,headwidth=10,scale=40,color="#444444")
plt.xlim([-2, 2]) #x축의 최대,최소값 정하기
plt.ylim([-2, 2]) #y축의 최대,최소값 정하기
plt.xlabel('x0')  #x축의 이름 정하기
plt.ylabel('x1')  #y축의 이름 정하기
plt.grid()
plt.legend(['x1'])   #그래프의 이름 정하기
plt.draw()
#plt.show()        #새로운 figure 보여주기기

"""
4.4.1 경사 하강법
"""

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x
    x_history = []

    for i in range(step_num):

        x_history.append(x.copy())

        grad = numerical_gradient(f,x)
        x -= lr*grad

    return x , np.array(x_history)

def function_a(x):
    return np.sum(x**2)

init_x = np.array([-3.0,4.0])
x, x_history = gradient_descent(function_a,init_x=init_x,lr = 0.1)

plt.figure()
plt.plot( [-5, 5], [0,0], '--b')
plt.plot( [0,0], [-5, 5], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-3.5, 3.5)
plt.ylim(-4.5, 4.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()

"""
4.4.2 신경망에서의 기울기
"""
from common.functions import *
from common.gradient import numerical_gradient

class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) #정규분포를 이용하여 초기화

    def predict(self, x):
        return np.dot(x,self.W)

    def loss(self,x,t):
        z = self.predict(x)
        y = softmax(x)
        loss = cross_entropy_error(y,t)

        return loss

net = SimpleNet()
print(net.W)

x = np.array([0.6, 0.9])
t = np.array([0, 1, 0])

f = lambda w: net.loss(x,t)
dW = numerical_gradient(f, net.W)

print(dW)
