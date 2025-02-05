from torch import nn


class MultiLayerPerceptronModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()  # 입력층: 1차원 벡터로 평탄화
        self.fc1 = nn.Linear(784, 256)  # 입력 뉴런: 784, 출력 뉴런: 256
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # 30% 확률로 뉴런을 0으로 만듦
        self.fc2 = nn.Linear(256, 10)  # 10개로 분류할거니깐

    def forward(self, data):
        data = self.flatten(data)
        data = self.fc1(data)
        data = self.relu(data)
        data = self.dropout(data)
        logits = self.fc2(data)  # logits: 출력값
        return logits
