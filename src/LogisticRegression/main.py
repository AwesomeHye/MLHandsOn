import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from src.LogisticRegression.model import LogisticRegressionModel

loaded_data = np.loadtxt('/Users/hiseo/DataSet/diabetes.csv', delimiter=',', skiprows=1) # numpy.ndarray
print(type(loaded_data))

x_train_np = loaded_data[:, 0:-1] # 학습 데이터
y_train_np = loaded_data[:, [-1]] # 정답 데이터

print(x_train_np.shape, y_train_np.shape)

x_train = torch.Tensor(x_train_np)
y_train = torch.Tensor(y_train_np)

print(x_train.shape, y_train.shape)

model = LogisticRegressionModel()
for param in model.parameters(): # 파라미터 초기값
    print(param)

loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001) # lr: learning rate. 얼마나 큰폭으로 숫자 바꿀건지.

train_loss_list = []
train_accuracy_list = []

nums_epoch = 5000 # epoch: 주기

for epoch in range(nums_epoch + 1):
    outputs = model(x_train)

    loss = loss_function(outputs, y_train)
    train_loss_list.append(loss.item())

    prediction = outputs > 0.5
    correct = (prediction.float() == y_train)
    accuracy = correct.sum().item() / len(correct)
    train_accuracy_list.append(accuracy)


    ## 모델 파라미터(가중치, 바이어스) 업데이트
    optimizer.zero_grad()  # 기울기(gradient) 초기화 (이전 값이 남아있지 않도록)
    loss.backward() # 역전파(Backpropagation) 수행 (기울기 계산)
    optimizer.step() # 계산된 기울기로 가중치 업데이트 (학습 진행)

    if epoch % 100 == 0:
        print('epoch = ', epoch, 'current loss = ', loss.item(), ' accuracy = ', accuracy)


plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(train_loss_list, label='train loss')
plt.legend(loc='best')

plt.show()


plt.title('Accuracy Trend')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()

plt.plot(train_accuracy_list, label='train accuracy')
plt.legend(loc='best')

plt.show()