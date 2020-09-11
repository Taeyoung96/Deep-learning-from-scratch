import numpy as np
import matplotlib.pyplot as plt
from Chapter7_1 import SimpleConvNet

"""
필터 계수 시각화 하기
"""
def filter_show(filters, nx=8, margin=3, scale=10):
    FN, C, FH, FW = filters.shape
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()  #피규어 생성
    fig.subplots_adjust(left=0, right=1, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i,0], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

network = SimpleConvNet()
#무작위 초기화 후의 가중치
filter_show(network.params['W1'])

#학습된 가중치
network.load_params("params.pkl")
filter_show(network.params['W1'])
