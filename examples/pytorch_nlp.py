import torch
from torch import nn
import sys
import logging
from logging import Logger
import math, random
from pathlib import Path

from typing import Union, Optional, List, Tuple, NamedTuple, Set, Dict, Callable
from typing import Generator, Iterable, KeysView, ValuesView, ItemsView, Any, Type, NewType

import numpy as np
from colorlog import colorlog

import torch
from torch import Tensor
from torch.nn import init
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, sampler
from torch import nn, cuda, initial_seed, autograd, optim, distributions
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pack_padded_sequence, pad_sequence, pad_packed_sequence

from torchtext.vocab import Vocab, SubwordVocab
from torchtext.data import Field, SubwordField, LabelField, NestedField, Dataset, Iterator, Example, Batch

from torchtext.datasets import SST
from torchtext.vocab import Vocab
from aku import Aku


class TokenEmbedding(nn.Embedding):
    def __init__(self, dim: int, *, vocab: Vocab):
        super(TokenEmbedding, self).__init__(
            num_embeddings=vocab.stoi.__len__(),
            embedding_dim=dim, padding_idx=vocab.stoi.get('<pad>', None),
        )


class Encoder(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, bias: bool, *, embedding_layer: nn.Embedding):
        super(Encoder, self).__init__()
        self.rnn = nn.LSTM(
            input_size=embedding_layer.embedding_dim,
            hidden_size=hidden_dim, bidirectional=True,
            batch_first=True, bias=bias,
            num_layers=num_layers, dropout=0.,
        )
        self.encoding_dim = hidden_dim * 2

    def forward(self, inputs):
        _, (h, _) = self.rnn(inputs)
        h = h.view(self.rnn.num_layers, 2, -1, self.rnn.hidden_size)[-1]
        return torch.cat([h[0], h[1]], dim=-1)


class OutputLayer(nn.Module):
    def __init__(self, *, num_targets, encoder_layer: Encoder):
        super(OutputLayer, self).__init__()
        self.fc = nn.Linear(
            encoder_layer.encoding_dim,
            num_targets,
        )

    def forward(self, inputs):
        return super(OutputLayer, self).forward(inputs)


class Model(nn.Module):
    def __init__(self, embedding_layer: (TokenEmbedding, ),
                 encoder_layer: (Encoder, ),
                 output_layer: (OutputLayer, ), *,
                 word_vocab: Vocab, target_vocab: Vocab):
        super(Model, self).__init__()
        self.embedding_layer = embedding_layer(
            vocab=word_vocab,
        )
        self.encoder_layer = encoder_layer(
            embedding_layer=self.embedding_layer,
        )
        self.output_layer = output_layer(
            encoder_layer=self.encoder_layer,
            num_targets=len(target_vocab),
        )

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=target_vocab.stoi.get('<pad>', None),
            reduction='mean',
        )

    def forward(self, batch):
        embedding = self.embedding_layer(batch.word)
        encoding = self.encoder_layer(embedding)
        outputs = self.output_layer(encoding)
        return outputs

    def fit(self, batch):
        outputs = self(batch)
        return self.criterion(outputs, batch.target)

    def transform(self, batch):
        outputs = self(batch)
        return self.criterion(outputs, batch.target), outputs


def sst(batch_size: int = 10):
    return SST.iters(batch_size=batch_size, device=torch.device('cpu'))


def sgd(lr: float = 1e-1, momentum: float = 0.0, *, model: nn.Module):
    return optim.SGD(
        params=[param for param in model.parameters() if param.requires_grad],
        lr=lr, momentum=momentum,
    )


def adam(lr: float = 1e-3, amsgrad: bool = False, *, model: nn.Module):
    return optim.Adam(
        params=[param for param in model.parameters() if param.requires_grad],
        lr=lr, amsgrad=amsgrad,
    )


app = Aku()


@app.register
def train(task: (sst, ), model: (Model, ), optimizer: (sgd, adam), num_epochs: int = 20, **kwargs):
    train, dev, test = task()
    word_vocab = train.dataset.fields['text'].vocab
    target_vocab = train.dataset.fields['label'].vocab
    model = model(word_vocab=word_vocab, target_vocab=target_vocab)
    print(f'model => {model}')
    optimizer = optimizer(model=model)
    print(f'optimizer => {optimizer}')

    for epoch in range(1, 1 + num_epochs):
        for batch in train:
            loss = model.fit(batch)

            model.zero_grad()
            loss.backward()
            print(f'loss => {loss.item():.4f}')
            optimizer.step()


if __name__ == '__main__':
    app.run()
