import gensim.utils
from gensim.models import Word2Vec, KeyedVectors

from src.Word2vecWithMlp.config import WORD2VEC_MODEL_NAME


class Word2vec:
    def __init__(self):
        # ["가게명 메뉴명1 메뉴명 메뉴명3"] 이렇게?
        sentences = [
            "I love machine learning",
            "Deep learning is amazing",
            "I enjoy natural language processing",
            "Neural networks are powerful",
            "Machine learning makes things smarter"
        ]

        tokenized_sentences = [gensim.utils.simple_preprocess(sentence) for sentence in sentences]

        self.word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=50, window=2, sg=0,
                                       min_count=1, workers=2)
        self.word2vec_model.save(WORD2VEC_MODEL_NAME)

    def print_vector(self, word):
        print(self.word2vec_model.wv[word])
