import numpy as np
import sys, os

from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict
from mnist import load_mnist

sys.path.append(os.pardir)

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        #가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

        #계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastlayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # x: 입력 데이터, t: 정답 데이터
    def loss(self, x, t):
        y = self.predict(x)

        return self.lastlayer.forward(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y , axis=1)
        if t.ndim != 1:
            t = np.argmax(t,axis=1)

        accuracy = np.sum(y == t) /  float(x.shape[0])

        return accuracy

    #가중치 매개변수의 기울기를 수치 미분 방식으로 구한다.
    def numerical_gradient(self,x,t):
        loss_w = lambda w: self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])

        return grads

    #가중치 매개변수의 기울기를 오차역전파법(backpropagation)으로 구한다.
    def gradient(self,x,t):
        #순전파
        self.loss(x,t)

        #역전파
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values()) #layers라는 이름의 리스트에 value값 순서대로 저장
        layers.reverse() #리스트의 순서 거꾸로하기
        for layer in layers:
            dout = layer.backward(dout)

        #결과 저장
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

if __name__ == '__main__':
    #데이터 읽기
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    """
    # backpropagation으로 구한 기울기 검증하기
    x_batch = x_train[:3]
    t_batch = t_train[:3]

    grad_numerical = network.numerical_gradient(x_batch,t_batch)
    grad_backprop = network.gradient(x_batch,t_batch)

    #각 가중치의 차이의 절대값을 구한 후, 그 절대값들의 평균을 낸다.
    for key in grad_numerical.keys():
        diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
        print(key +" : "+str(diff))
        

    수치 미분과 오차 역전파법의 결과가 딱 0이되는 일은 드물다.
    컴퓨터가 할 수 있는 계산은 유한하기 때문이다.
    하지만 결과로 0의 거의 근접한 숫자가 나오긴 한다.
    """

    iters_num = 10000
    train_size = x_train.shape[0] #60000
    batch_size = 100
    learning_rate = 0.1

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size/ batch_size,1)

    for i in range(iters_num):
        batch_mask = np.random.choice(train_size,batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        #backpropagation으로 기울기 구하기
        grad = network.gradient(x_batch, t_batch)

        #갱신
        for key in ('W1','b1','W2','b2'):
            network.params[key] -= learning_rate * grad[key]

            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train,t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train accuracy :",train_acc,"test accuracy : ",test_acc)
