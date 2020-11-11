from typing import Type, Union
from aku import Aku


class Embedding(object):
    def __init__(self, num_words: int = 30000, word_dim: int = 100,
                 num_chars: int = 1000, char_dim: int = 50):
        super(Embedding, self).__init__()
        print(f'num_words => {num_words}')
        print(f'word_dim => {word_dim}')
        print(f'num_chars => {num_chars}')
        print(f'char_dim => {char_dim}')

        self.embedding_dim = word_dim + char_dim


class BiLSTMEncoder(object):
    def __init__(self, input_dim: int = 100, hidden_dim: int = 200, bidirectional: bool = True, *, encoding_dim: int):
        super(BiLSTMEncoder, self).__init__()
        print(f'input_dim => {input_dim}')
        print(f'hidden_dim => {hidden_dim}')
        print(f'bidirectional => {bidirectional}')
        print(f'encoding_dim => {encoding_dim}')


class TransformerEncoder(object):
    def __init__(self, q_dim: int = 100, k_dim: int = 100,
                 v_dim: int = 100, model_dim: int = 20, num_heads: int = 5, *, encoding_dim: int):
        super(TransformerEncoder, self).__init__()
        print(f'q_dim => {q_dim}')
        print(f'k_dim => {k_dim}')
        print(f'v_dim => {v_dim}')
        print(f'model_dim => {model_dim}')
        print(f'num_heads => {num_heads}')
        print(f'encoding_dim => {encoding_dim}')


aku = Aku()


@aku.option
def one(embedding: Type[Embedding],
        encoder: Union[Type[BiLSTMEncoder], Type[TransformerEncoder]]):
    embedding = embedding()
    encoder = encoder(encoding_dim=embedding.embedding_dim)


if __name__ == '__main__':
    aku.run()
