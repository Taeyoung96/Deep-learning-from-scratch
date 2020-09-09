import numpy as np
import os
import sys
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import matplotlib.pyplot as plt
from mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet

"""
SGD 구현
"""
class SGD:
    def __init__(self, lr = 0.01):
        self.lr = lr

    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

"""
Momentum 구현
"""
class Momentum:
    def __init__(self,lr = 0.01, momentum = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self,params,grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            #속도를 모멘텀과 기울기를 이용하여 update
            params[key] += self.v[key]
"""
AdaGrad 구현
"""
class AdaGrad:
    def __init__(self, lr = 0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self,params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / np.sqrt(self.h[key] +1e-7) #0이 되는 것을 막아주기 위해서

"""
Adam 구현
"""
class Adam:
    def __init__(self, lr = 0.001,beta1 =0.9, beta2 = 0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

if __name__ == "__main__":
    # 0. MNIST DATA 읽기
    (x_train, t_train),(x_test, t_test) = load_mnist(normalize=True)
    train_size = x_train.shape[0]
    batch_size = 128
    max_iterations = 2000

    # 1. 실험용 설정
    optimizers = {}
    optimizers['SGD'] = SGD()
    optimizers['Momentum'] = Momentum()
    optimizers['AdaGrad'] = AdaGrad()
    optimizers['Adam'] = Adam()

    networks = {}
    train_loss = {}
    for key in optimizers.keys():
        networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100], output_size=10)
        train_loss[key] = []

    # 2. 훈련 시작
    for i in range(max_iterations):
        batch_mask = np.random.choice(train_size,batch_size) # batchsize = (1, 128)
        x_batch = x_train[batch_mask]                        # x_batch = (128,784)
        t_batch = t_train[batch_mask]                        # t_batch = (1, 128)
        for key in optimizers.keys():
            grads = networks[key].gradient(x_batch, t_batch)
            optimizers[key].update(networks[key].params, grads)

            loss = networks[key].loss(x_batch,t_batch)
            train_loss[key].append(loss)

        if i % 100 == 0:
            #print("===========" + "iteration:" + str(i) + "===========")
            for key in optimizers.keys():
                loss = networks[key].loss(x_batch,t_batch)
                #print(key + ":"+str(loss))

    markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
    x = np.arange(max_iterations)
    for key in optimizers.keys():
        plt.plot(x,smooth_curve(train_loss[key]),marker = markers[key], markevery = 100, label=key)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.ylim(0,1)
    plt.legend()
    plt.show()


