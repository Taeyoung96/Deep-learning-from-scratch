import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


"""
단순한 합성곱 신경망 구현하기

conv - relu - pool - affine - relu - affine - softmax
    
    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
"""

class SimpleConvNet:
    def __init__(self, input_dim = (1,28,28),
                 conv_param={'filter_num':30,'filter_size':5,'pad':0,'stride':1},
                 hidden_size = 100, output_size = 10, weight_init_std = 0.01):
        #hidden_size = 은닉층(완전 연결) 뉴런의 수
        #output_size = 출력층(완전 연결) 뉴런의 수
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1] #어차피 높이하고 너비 같으므로 둘 중 하나 가져와도 됨
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        #가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.rand(filter_num, input_dim[0],filter_size, filter_size)
                                                           #(필터의 갯수 N, 입력 채널 수 C, 필터 높이 FH, 필터 너비 FW)
        self.params['b1'] = np.zeros(filter_num)  #필터의 갯수 = output channel의 갯수
        self.params['W2'] = weight_init_std * np.random.rand(pool_output_size,hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        #계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    #첫 계층부터 마지막 계층까지 순차적으로 진행
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self,x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size = 100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size : (i+1)*batch_size]
            tt = t[i*batch_size : (i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y,axis=1)
            acc += np.sum(y == tt)  #y와 tt중에 같은 위치에 같은 값을 가진 것이 몇 개인지 check

        return acc / x.shape[0]

    def gradient(self, x, t):
        #forward
        self.loss(x,t)

        #backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        #결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

    def save_params(self, file_name='params.pkl'):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params,f)

    def load_params(self, file_name = "params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)

        #params.pkl에 있는 파라미터의 값들로 params의 값을 update
        for key, val in params.items():
            self.params[key] = val

        #각각 w,b값들을 update
        for i, key in enumerate(['Conv1', 'Affine1','Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
