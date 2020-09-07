from common.functions import *
from common.gradient import numerical_gradient
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import sys,os
sys.path.append(os.pardir)
from mnist import load_mnist

"""
4.5 학습 알고리즘 구현하기
"""
class TwoLayerNet:
    def __init__(self,input_size, hidden_size, output_size, weight_init_std = 0.01):

        #가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)

        return y

    def loss(self,x,t):
        y = self.predict(x)

        return cross_entropy_error(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)

        accuracy = np.sum(y==t) / float(x.shape[0])
        return  accuracy

    # x : 입력 데이터, t : 정답 테이블
    def numerical_gradient(self,x,t):
        loss_w = lambda w : self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_w, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_w, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_w, self.params['b2'])

        return grads


"""
미니배치 학습 구현하기
"""

(x_train, t_train),(x_test,t_test) = load_mnist(normalize=True,one_hot_label=True)
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
train_loss_list = []
train_acc_list = []
test_acc_list = []

#하이퍼 파라미터
iters_num = 100
train_size = x_train.shape[0]
batch_size = 100 # 미니 배치의 크기
learning_rate = 0.1

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)
print(iter_per_epoch)

for i in range(iters_num):
    #미니 배치 얻어오기
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    #기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)

    #매개변수 갱신
    for key in {'W1', 'b1', 'W2', 'b2'}:
        network.params[key] -= learning_rate * grad[key]

    #학습 경과 기록
    loss = network.loss(x_batch,t_batch)
    train_loss_list.append(loss)

    print("현재",i,"번째 입니다.")

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

"""
iter_per_epoch를 계산해보면 600인데
600번 마다 test를 진행하고 그 값을 추가해준 코드인 것 같다.
하지만 직접 해보면 알겠지만 시간이 너무 오래걸린다...
"""