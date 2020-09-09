import numpy as np
import matplotlib.pyplot as plt

import sys,os
sys.path.append(os.pardir)
import pickle
#sys.path.append('/home/taeyoungkim/PycharmProjects') #절대 경로를 이용하여 경로 추가
from mnist import load_mnist                         #MNIST dataset load
from PIL import Image


"""
Input X 생성
"""
x = np.arange(-5.0,5.0,0.1)
#arange 함수는 반열린구간 [start, stop) 에서 step 의 크기만큼 일정하게 떨어져 있는 숫자들을 array 형태로 반환해 주는 함수

"""
3.2.3 계단 함수의 그래프 만들기
"""
def step_function(x):
    return np.array(x > 0 ,dtype=np.int)

y = step_function(x)
plt.plot(x,y,label='step function') #그래프 그리고 Label 달기

"""
3.2.4 시그모이드 함수 구현하기
"""

def sigmoid(x):
    x_float = x.astype('float128')
    #RuntimeWarning: overflow encountered in exp  - exp 오버플로우를 방지하기 위해 형변

    return 1 / (1+np.exp(-x_float))
# 넘파이의 브로드캐스팅때문에 넘파이 배열로도 충분히 스칼라값과 배열의 원소 연산이 가능하다.

z = sigmoid(x)

plt.plot(x,z,label='Sigmoid')  #그래프 그리고 Label 달기

"""
3.2.7 ReLU 함수
"""
def ReLU(x):
    return np.maximum(0,x)

t = ReLU(x)
plt.plot(x,t,label='ReLU')  #그래프 그리고 Label 달기

"""
그래프 출력
"""
# plt.legend()          #라벨들 그래프 출력할 때 보이도록 하
# plt.ylim(-0.1,1.1)  #y의 범위 정해주기
# plt.show()            #그래프 출력

"""
3.4.2
다차원 배열로 신경망 구성하기
A = XW + B
"""
def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network

def forward(network,x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = a3 # y= identity_function(a3)


    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)

"""
3.5.1
softmax 함수 구현하기

"""

def softmax(a):
    """
    원래 식은 이게 맞지만 오버플로우 문제 때문에 수식을 개선해서 사용한다.
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    """
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

"""
3.6
Mnist 데이터셋을 이용하여 손글씨 숫자 인식하기
"""


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))  #이미지 읽으려고 PIL 이미지로 변
    pil_img.show()

(x_train,y_train), (x_test,y_test) = load_mnist(flatten=True, normalize=False)

# load_mnist 함수는 3가지 인수를 결정해 주는데 모두 bool값
# 첫번째는 flatten -> 1차원 배열로 만들어 줄 것인가?
# 두번째는 normalize ->정규화를 0~1사이로 시킬 것인가?  아니면 0 ~ 255 그대로 사용할 것인가?
# 세번째는 one_hot_label -> 원 핫 인코딩을 할 것인가?


print(x_train.shape)  #Mnist가 27*27*1 의 이미지 인데 일차원으로 flatten 했으므로 열이 784이다.

img = x_train[0]
label = y_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28)  #이미지를 표시할 때는 다시 원래 형상으로 바꿔주어야 한다.

img_show(img)


"""
3.6.2 신경망의 추론 처리
"""

def get_data():
    (x_train,y_train), (x_test,y_test) = load_mnist(flatten=True, normalize=False)

    return x_test,y_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

batch_size = 100 # 배치 크기
accuracy_cnt = 0
for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch,axis=1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])


print("Accuracy: "+str(float(accuracy_cnt)/len(x)))




