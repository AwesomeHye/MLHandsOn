import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from model import MultiLayerPerceptronModel
from datetime import datetime
import matplotlib.pyplot as plt

# input: 손글씨로 숫자가 그려진 이미지 벡터
# output: 0~9 까지의 손글씨로 쓴 숫자
def model_train(dataloader, model, loss_function, optimizer):
    model.train()  ## 신경망을 학습 모드로 전환 (모델 파라미터 업데이트 가능)
    train_loss_sum = train_correct = train_total = 0
    total_train_batch = len(dataloader)

    for images, labels in dataloader:  # images: 이미지, labels: 정답 숫자 0~9
        x_train = images.view(-1, 28 * 28)  # 28*28 2차원 이미지를 784 차원 벡터로 변환, Tensor(32, 784) -> 이미지 32개씩 784 차원
        y_train = labels  # Tesnor(32) -> 이미지 32개에 대한 정답 32 개

        outputs = model(x_train)  # Tensor(32, 10) -> 이미지 32개에 대한 10개의 확률
        loss = loss_function(outputs, y_train)  # tensor(2.3663) -> 손실함수 값이 2.3663 인 스칼라값

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()  # Tensor.item() -> 2.3663 숫자로 변환
        train_total += y_train.size(0)  #  Tesnor(32) 에서 첫 번째 차원인 32 반환
        train_correct += ((torch.argmax(outputs, 1)) == y_train).sum().item()

        # torch.argmax(outputs, 1): Tensor(32, 10) 에서 각 행(row)마다 10개의 값(인덱스 [1]) 중에서 가장 큰 값의 인덱스를 찾아 반환
        # (torch.argmax(outputs, 1)) == y_train 각 행 비교해서 boolean tensor 생성 [F, T, ..10개]
        # ((torch.argmax(outputs, 1)) == y_train).sum() 서로 같은 개수 tensor(2) 반환
        # .item() 값으로 바꿔줌 -> 2

    train_avg_loss = train_loss_sum / total_train_batch
    train_avg_accuracy = 100 * train_correct / train_total

    return train_avg_loss, train_avg_accuracy


def model_evaluate(dataloader, model, loss_function):
    model.eval()  # 신경망을 추론(검증)모드로 전환

    with torch.no_grad():  # 미분 연산 안함. 이거 쓰면 메모리 소비 감소
        val_loss_sum = val_correct = val_total = 0
        total_val_batch = len(dataloader)

        for images, labels in dataloader:
            x_val = images.view(-1, 28 * 28)
            y_val = labels

            outputs = model(x_val)
            loss = loss_function(outputs, y_val)

            val_loss_sum += loss.item()
            val_total += y_val.size(0)
            val_correct += ((torch.argmax(outputs, 1)) == y_val).sum().item()

        val_avg_loss = val_loss_sum / total_val_batch
        val_avg_accuracy = 100 * val_correct / val_total

    return val_avg_loss, val_avg_accuracy


def model_test(dataloader, model, loss_function):
    test_avg_loss, test_avg_accuracy = model_evaluate(dataloader, model, loss_function)

    print('accuracy:', test_avg_accuracy)
    print('loss:', test_avg_loss)


################## 데이터 불러오기 시작 ##################

## train 6만개 test 1만개 다운로드, train 의 15%는 검증에 쓰인다.
train = datasets.MNIST(root='/Users/hiseo/DataSet/MNIST', train=True,
                       transform=transforms.ToTensor(),  # 이미지는 0~255 사이의 픽셀값이라 0~1 사이의 값으로 정규화해줌
                       download=False)

test = datasets.MNIST(root='/Users/hiseo/DataSet/MNIST', train=False, transform=transforms.ToTensor(), download=False)

train_size = int(len(train) * 0.85)
validation_size = int(len(train) * 0.15)
train, validation = random_split(train, [train_size, validation_size])
print('train:', len(train), 'validation:', len(validation), 'test:', len(test))  # train: 51000 validation: 9000 test: 10000

################## 데이터 읽기 ##################

BATCH_SIZE = 32
train_loader = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(dataset=validation, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test, batch_size=BATCH_SIZE, shuffle=True)

model = MultiLayerPerceptronModel()
loss_function = nn.CrossEntropyLoss()  # crossEntropy 에 softmax 포함되어있다.
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)  # 모델의 가중치(W)와 편향(b)를 옵티마이저가 업데이트할 수 있도록 파라미터 전달

train_loss_list = []
train_accuracy_list = []

val_loss_list = []
val_accuracy_list = []

EPOCHS = 20
start_time = datetime.now()

for epoch in range(EPOCHS):
    train_avg_loss, train_avg_accuracy = model_train(train_loader, model, loss_function, optimizer)
    train_loss_list.append(train_avg_loss)
    train_accuracy_list.append(train_avg_accuracy)

    val_avg_loss, val_avg_accuracy = model_evaluate(validation_loader, model, loss_function)
    val_loss_list.append(val_avg_loss)
    val_accuracy_list.append(val_avg_accuracy)

    print('epoch:', '%02d' % (epoch + 1),
          'train loss =', '{:.4f}'.format(train_avg_loss), 'train accuracy =', '{:.4f}'.format(train_avg_accuracy),
          'validation loss =', '{:.4f}'.format(val_avg_loss), 'validation accuracy =',
          '{:.4f}'.format(val_avg_accuracy))

end_time = datetime.now()

print('elapsed time => ', end_time - start_time)

model_test(test_loader, model, loss_function)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Loss Trend')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid()

plt.plot(train_loss_list, label='train loss')
plt.plot(val_loss_list, label='validation loss')
plt.legend()



plt.subplot(1, 2, 2)
plt.title('Accuracy Trend')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.grid()
plt.plot(train_accuracy_list, label='train accuracy')
plt.plot(val_accuracy_list, label='validation accuracy')
plt.legend()

plt.show()