import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *

"""
좀 더 깊은 합성곱 신경망
네트워크 구성은 아래와 같음
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
특징
- 3*3의 작은 필터를 사용한 합성곱 계층
- 활성화 함수는 Relu
- 완전연결 계층(Fully connected layer) 뒤에 dropout layer 사용
- Adam을 이용하여 최적화
- 가중치 초기값은 : 'He 초기값'
"""
class DeepConvNet:
    def __init__(self, input_dim = (1,28,28),
                 conv_param_1 = {'filter_num':16, 'filter_size':3, 'pad':1, 'stride':1},
                 conv_param_2 = {'filter_num':16, 'filter_size':3,'pad':1,'stride':1},
                 conv_param_3 = {'filter_num':32, 'filter_size':3,'pad':1, 'stride':1},
                 conv_param_4 = {'filter_num':32, 'filter_size':3,'pad':2, 'stride':1},
                 conv_param_5 = {'filter_num':64, 'filter_size':3,'pad':1, 'stride':1},
                 conv_param_6 = {'filter_num':64, 'filter_size':3,'pad':1, 'stride':1},
                 hidden_size = 50, output_size = 10):

        #가중치 초기화====================================
        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums) # Relu를 사용하기 때문에 He로 초기화 맞춤

        self.params = {}
        pre_channel_num = input_dim[0]  # (28,28)

        #conv 필터 초기화
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5,
                                          conv_param_6]):
            self.params['W'+str(idx+1)] = weight_init_scales[idx] * np.random.randn(conv_param['filter_num'],\
                                          pre_channel_num,conv_param['filter_size'], conv_param['filter_size'])
            self.params['b'+str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        #hidden layer 초기화
        self.params['W7'] = weight_init_scales[6] * np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        #output layer 초기화
        self.params['W8'] = weight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)


        #계층 생성====================================
        self.layers = []
        #conv + relu
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'],conv_param_1['pad']))
        self.layers.append(Relu())
        #conv + relu
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'],conv_param_2['pad']))
        self.layers.append(Relu())
        #max pooling
        self.layers.append(Pooling(pool_h=2,pool_w=2,stride=2))
        #conv + relu
        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        # conv + relu
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        # max pooling
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        # conv + relu
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                                       conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        # conv + relu
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                                       conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        # max pooling
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        # affine - fully connected layer + relu (hidden layer)
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        #drop out layer
        self.layers.append(Dropout(0.5))
        # affine - fully connected layer + relu (output layer)
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))

        #Softmax layer
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg = False):
        for layer in self.layers:
            if isinstance(layer, Dropout):  #layer가 dropout인지 확인
                x = layer.forward(x, train_flg)  #맞으면 train 안함.
            else:
                x = layer.forward(x)

        return x

    def loss(self,x,t):
        y = self.predict(x,train_flg=True)
        return self.last_layer.forward(y,t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self,x,t):
        #forward
        self.loss(x,t)

        #backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        #결과 저장
        grads = {}
        for i, layer_idx in enumerate((0,2,5,7,10,12,15,18)):  #conv, affine layer만
            grads['W'+str(i+1)] = self.layers[layer_idx].dW
            grads['b'+str(i+1)] = self.layers[layer_idx].db

        return grads

    #파라미터 값 저장하기
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    #파라미터 값 불러오기
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i + 1)]
            self.layers[layer_idx].b = self.params['b' + str(i + 1)]