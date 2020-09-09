import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def ReLU(x):
    return np.maximum(0,x)

def tanh(x):
    return np.tanh(x)

input_data = np.random.randn(1000,100) #1000개의 데이터
node_num = 100
hidden_layer_size = 5
activations = {} #이곳에 활성화 함수 결과를 저장

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    #초기값을 다양하게 설정해보자
    #w = np.random.randn(node_num,node_num) * 1
    #w = np.random.randn(node_num,node_num) * 0.01
    #w = np.random.randn(node_num,node_num) * np.sqrt(1.0 / node_num) #Xavier 초기값
    w = np.random.randn(node_num,node_num) * np.sqrt(2.0 / node_num)  #He초기값

    a = np.dot(x, w)

    #활성화 함수
    #z = sigmoid(a)
    #z = ReLU(a)
    z = tanh(a)

    activations[i] = z

#히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1,len(activations), i+1)  #1*len(activations) 형태로 순서대로 subplot을 만든다.
    plt.title(str(i+1)+"-layer")          #subplot의 제목 쓰는 것
    if i != 0:
        plt.yticks([],[])
    plt.hist(a.flatten(),30,range=(0,1))

plt.show()