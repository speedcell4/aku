from abc import ABCMeta, abstractmethod
from typing import Type
from typing import Union, List, Tuple

import torch
from einops import rearrange
from torch import Tensor
from torch import nn, optim
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from torchglyph.vocab import Vocab

from aku import Literal


class Corpus(object):
    @classmethod
    def new(cls, batch_size: int) -> None:
        raise NotImplementedError


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
    def __init__(self, hidden_dim: int = 300, num_layers: int = 1, *, embedding_layer: WordEmbedding) -> None:
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
    def __init__(self, kernel_sizes: Tuple[int, ...] = (3, 5, 7), hidden_dim: int = 200, *,
                 embedding_layer: WordEmbedding) -> None:
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
                 reduction: Literal['sum', 'mean'] = 'mean', *,
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

    def fit(self, word: PackedSequence, target: Tensor) -> Tensor:
        projection = self(word)
        return self.criterion(projection, target)

    def inference(self, word: PackedSequence) -> List[int]:
        projection = self(word)
        return projection.detach().cpu().argmax(dim=-1).tolist()


class SGD(optim.SGD):
    def __init__(self, lr: float = 1e-3, momentum: float = 0, dampening: float = 0,
                 weight_decay: float = 0, nesterov: bool = False, *, module: nn.Module) -> None:
        super(SGD, self).__init__(
            params=module.parameters(),
            lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov,
        )


class Adam(optim.Adam):
    def __init__(self, lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0, amsgrad: bool = False, *, module: nn.Module) -> None:
        super(Adam, self).__init__(
            params=module.parameters(),
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad,
        )


def train_classifier(
        num_epochs: int = 100,
        Data: Type[Corpus.new] = Corpus.new,
        Cls: Type[TextClassifier] = TextClassifier,
        Opt: Type[Union[SGD, Adam]] = Adam,
):
    train, dev, test = Data()
    classifier = Cls(word_vocab=..., target_vocab=...)
    optimizer = Opt(module=classifier)

    for epoch in range(1, num_epochs + 1):
        classifier.train()
        for batch in train:
            loss = classifier(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
