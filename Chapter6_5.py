import os
import sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer

"""
MNIST 데이터셋으로 본 dropout을 이용하여 오버피팅을 막는 방법
신경망 구조가 복잡할 때 효과적
"""

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 오버피팅을 재현하기 위해 학습 데이터 수를 줄임
x_train = x_train[:300]
t_train = t_train[:300]

#드랍아웃 사용 유무와 비율 설정===========
use_dropout = True
dropout_ratio = 0.2
#x_shpae이랑 같은 모양 중에서 랜덤하게 dropout_ratio보다 작은 값으로 예측 한 것들은 무시해 버리겠다..
#====================================

network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100,100,100,100,100,100],
                              output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)
trainer = Trainer(network,x_train,t_train,x_test,t_test,
                  epochs=301, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr':0.01}, verbose=True)  #verbose는 중간중간 출력값을 나타내고 싶을 때

trainer.train()  #훈련 시작!

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

#그래프 그리기=======================
markers = {'trainer':'o','test':'s'}
x = np.arange(len(train_acc_list))

plt.plot(x,train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x,test_acc_list, marker='s', label='test', markevery=10)

plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0,1.0)
plt.legend(loc="lower left")
plt.show()

