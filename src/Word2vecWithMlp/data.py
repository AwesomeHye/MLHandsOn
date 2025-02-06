import torch
from torch.utils.data import Dataset
import numpy as np
from config import labels_dict


def sentence_to_vector(sentence, model, embedding_dim):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]  # 문장에 들어있는 단어 벡터 다 모으기
    return np.mean(vectors, axis=0) if vectors else np.zeros(
        embedding_dim)  # 단어 벡터들은 여러개지만 벡터평균 내서 한 문장 당 하나의 벡터로 만드나봄


class ShopDataSet(Dataset):

    def __init__(self, data, model, embedding_dim):  # embedding_dim: word2vec 만들 떄 설정한 임베딩 차원 크기
        self.data = data
        self.model = model

        self.x = torch.tensor([sentence_to_vector(text, model, embedding_dim) for text, _ in data],
                              dtype=torch.float32)
        self.y = torch.tensor([labels_dict[label] for _, label in data], dtype=torch.long)

    def __len__(self):
        return len(self.data)  # 한 배치 당 데이터 크기

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
