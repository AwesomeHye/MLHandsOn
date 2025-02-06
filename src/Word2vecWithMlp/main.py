import torch
from gensim.models import Word2Vec
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from data import ShopDataSet, sentence_to_vector

from src.Word2vecWithMlp.config import WORD2VEC_MODEL_NAME, labels_dict
from src.Word2vecWithMlp.model import MlpClassifier

# word2vec = Word2vec()
# word2vec.print_vector('machine')

## 학습 ##
data = [
    ("machine learning is powerful", "Tech"),
    ("deep learning improves AI", "AI"),
    ("natural language processing is cool", "NLP"),
    ("AI is transforming the world", "AI"),
    ("Neural networks are useful for NLP", "NLP")
]
word_vec_model = Word2Vec.load(WORD2VEC_MODEL_NAME)
embedding_dim = word_vec_model.vector_size

dataset = ShopDataSet(data, word_vec_model, embedding_dim)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

input_dim = embedding_dim
hidden_dim = 32
output_dim = len(labels_dict)

model = MlpClassifier(input_dim, hidden_dim, output_dim)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 뭔지 찾고 왜 SGD 대신 쓰는지 찾기

EPOCHS = 20
for epoch in range(EPOCHS):
    total_loss = 0
    for text, label in dataloader:
        outputs = model(text)
        loss = loss_function(outputs, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")

## 추론 ##
model.eval()

test_sentences = [
    "machine learning is evolving",
    "AI will change the world",
    "I love working on NLP projects"
]

for sentence in test_sentences:
    vector = sentence_to_vector(sentence, word_vec_model, embedding_dim)
    input_tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)  # unsqueeze(0): 샘플 1개라 첫 열에 샘플 수 뜻하는 1 추가
    with torch.no_grad():
        output = model(input_tensor)
    predicted_label_idx = torch.argmax(output, dim=1).item()
    predicted_label = list(labels_dict.keys())[predicted_label_idx]
    print(f"Input: {sentence}, Predicted: {predicted_label}")
