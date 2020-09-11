import os, sys
sys.path.append(os.pardir) #부모 디렉터리 파일을 가져올 수 있도록 설정

import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD

"""
MNIST 데이터셋으로 본 weight decay를 이용하여 오버피팅을 막는 방법
신경망이 단순할 때 효과적
"""

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

#오버피팅을 위해 학습 데이터 수를 줄이자.
x_train = x_train[:300]
t_train = t_train[:300]

#weight decay (가중치 감쇠) 설정
weight_decay_lambda = 0 # weight decay를 사용하지 않을 경우
#weight_decay_lambda = 0.1
#============================================

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size= 10
                        ,weight_decay_lambda =weight_decay_lambda)

optimizer = SGD(lr=0.01)

max_epochs = 201
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size/batch_size,1)  #여기서는 600
epoch_cnt = 0

for i in range(1000000):
    batch_mask = np.random.choice(train_size,batch_size)
    #train_size의 표본 중에서 batch_size 만큼 선택 batch_mask.shape = (1,100)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch) #기울기 계산
    optimizer.update(network.params,grads) #loss로 weight값 update

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train,t_train)
        test_acc = network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("epoch:",str(epoch_cnt),", train acc:",str(train_acc),", test acc:",str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break

#그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery = 10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery = 10)

plt.xlabel("epochs")
plt.ylabel("accuracy")

plt.ylim(0, 1.0)
plt.legend(loc = 'lower right')
plt.show()



