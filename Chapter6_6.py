import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.util import shuffle_dataset
from common.trainer import Trainer

"""
MNIST 데이터셋으로 검증셋을 이용하여 하이퍼파라미터의 최적화 구현
"""

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 결과를 빠르게 얻기 위해 훈련 데이터를 줄임
x_train = x_train[:500]
t_train = t_train[:500]

#20%로 검증 데이터 분할
validation_rate = 0.2
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)  #데이터셋을 무작위로 섞어준다.

x_val = x_train[:validation_num]
t_val = t_train[:validation_num]

x_train = x_train[validation_num:] #슬라이싱으로 검증 데이터셋 이후부터 훈련 데이터셋으로 사용
t_train = t_train[validation_num:]

def __train(lr, weight_decay, epocs = 50):
    #네트워크 생성
    network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size= 10, weight_decay_lambda=weight_decay)

    #훈련 초기화
    trainer = Trainer(network,x_train,t_train,x_test,t_test,epocs,mini_batch_size=100,
                      optimizer='adam', optimizer_param={'lr':lr},verbose=False)

    #훈련 시작
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


#하이퍼 파라미터 무작위 탐색================================
optimization_trial = 100
results_val = {}
results_train = {}

for _ in range(optimization_trial):
    #탐색한 하이퍼 파라미터의 범위 지정========================
    weight_decay = 10 ** np.random.uniform(-8,4)
    lr = 10 ** np.random.uniform(-6,-2)
    #===================================================

    val_acc_list, train_acc_list = __train(lr,weight_decay)
    print("val acc:",str(val_acc_list[-1]),"| lr:",lr,", weight decay:",str(weight_decay))
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

#그래프 그리기===============================================
print("=================Hyper-Parameter Optimization Result ===================")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num/ col_num))  #np.ceil은 올림 함수
i = 0

for key, val_acc_list in sorted(results_val.items(), key = lambda x:x[1][-1], reverse=True):
    print("Best-"+str(i+1)+"(val acc:"+str(val_acc_list[-1])+") | "+key)

    plt.subplot(row_num, col_num, i+1)
    plt.title("Best-"+str(i+1))
    plt.ylim(0.0,1.0)

    if i % 5:
        plt.yticks([])

    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.show()