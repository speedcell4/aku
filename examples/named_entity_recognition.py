from abc import ABCMeta, abstractmethod
from typing import List, Type, Union

import torch
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torchglyph.vocab import Vocab


class WordEmbedding(nn.Embedding):
    def __init__(self, *, word_vocab: Vocab) -> None:
        super(WordEmbedding, self).__init__(
            num_embeddings=len(word_vocab),
            embedding_dim=word_vocab.vec_dim,
            padding_idx=word_vocab.pad_idx,
            _weight=word_vocab.vectors,
        )

    def forward(self, word: PackedSequence) -> PackedSequence:
        data = super(WordEmbedding, self).forward(word.data)
        return word._replace(data=data)


class Encoder(nn.Module, metaclass=ABCMeta):
    encoding_dim: int

    def __init__(self, *, embedding_layer: WordEmbedding) -> None:
        super(Encoder, self).__init__()

    @abstractmethod
    def forward(self, embedding: PackedSequence) -> Tensor:
        raise NotImplementedError


class LstmEncoder(Encoder):
    def __init__(self, hidden_dim: int, num_layers: int, *, embedding_layer: WordEmbedding) -> None:
        super(LstmEncoder, self).__init__(embedding_layer=embedding_layer)
        self.rnn = nn.LSTM(
            input_size=embedding_layer.embedding_dim,
            hidden_size=hidden_dim, num_layers=num_layers,
            bias=True, batch_first=True, bidirectional=True,
        )

        self.encoding_dim = self.rnn.hidden_size * (2 if self.rnn.bidirectional else 1)

    def forward(self, embedding: PackedSequence) -> Tensor:
        _, (hidden, _) = self.rnn(embedding)
        return rearrange(hidden, '(l d) b h -> l b (d h)', l=self.rnn.num_layers)[-1]


class ConvEncoder(Encoder):
    def __init__(self, kernel_sizes: List[int], hidden_dim: int, *, embedding_layer: WordEmbedding):
        super(ConvEncoder, self).__init__(embedding_layer=embedding_layer)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1, out_channels=hidden_dim,
                    kernel_size=(kernel_size, embedding_layer.embedding_dim),
                    padding=(kernel_size // 2, 0), bias=True,
                ),
                nn.AdaptiveMaxPool2d(output_size=(1, 1)),
                nn.ReLU(),
            )
            for kernel_size in kernel_sizes
        ])

        self.encoding_dim = hidden_dim * len(kernel_sizes)

    def forward(self, embedding: PackedSequence) -> Tensor:
        data, _ = pad_packed_sequence(embedding, batch_first=True)
        return torch.cat([conv(data[:, None, :, :])[:, :, 0, 0] for conv in self.convs], dim=-1)


class Projection(nn.Sequential):
    def __init__(self, *, target_vocab: Vocab, encoding_layer: Encoder) -> None:
        super(Projection, self).__init__(
            nn.Linear(encoding_layer.encoding_dim, encoding_layer.encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_layer.encoding_dim, len(target_vocab)),
        )


class TextClassifier(nn.Module):
    def __init__(self,
                 Emb: Type[WordEmbedding] = WordEmbedding,
                 Enc: Type[Union[LstmEncoder, ConvEncoder]] = LstmEncoder,
                 Proj: Type[Projection] = Projection,
                 reduction: str = 'mean', *,
                 word_vocab: Vocab, target_vocab: Vocab) -> None:
        super(TextClassifier, self).__init__()

        self.embedding_layer = Emb(word_vocab=word_vocab)
        self.encoding_layer = Enc(embedding_layer=self.embedding_layer)
        self.projection_layer = Proj(target_vocab=target_vocab, encoding_layer=self.encoding_layer)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=target_vocab.pad_idx,
            reduction=reduction,
        )

    def forward(self, word: PackedSequence) -> Tensor:
        embedding = self.embedding_layer(word)
        encoding = self.encoding_layer(embedding)
        return self.projection_layer(encoding)

    def fit(self, word: PackedSequence, target: PackedSequence) -> Tensor:
        projection = self(word)
        return self.criterion(projection.data, target.data)

    def inference(self, word: PackedSequence) -> List[List[int]]:
        predictions, lengths = pad_packed_sequence(self(word), batch_first=True)
        predictions = predictions.detach().cpu().tolist()
        lengths = lengths.detach().cpu().tolist()
        return [
            predictions[i][:l]
            for i, l in enumerate(lengths)
        ]
