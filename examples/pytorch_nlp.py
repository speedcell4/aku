from typing import Type, Union

import torch
import torchtext
from torch import nn, optim
from torchtext.data import Field, Iterator
from torchtext.vocab import Vocab

from aku import Aku


class SST(torchtext.datasets.SST):
    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        TEXT = Field(batch_first=True)
        LABEL = Field(batch_first=True, sequential=False)

        train, val, test = cls.splits(TEXT, LABEL, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)

        return Iterator.splits(
            (train, val, test), device=device,
            batch_size=batch_size, sort=False, sort_within_batch=True,
        )


class TokenEmbedding(nn.Embedding):
    def __init__(self, dim: int = 100, *, vocab: Vocab):
        super(TokenEmbedding, self).__init__(
            num_embeddings=vocab.stoi.__len__(),
            embedding_dim=dim, padding_idx=vocab.stoi.get('<pad>', None),
        )


class Encoder(nn.Module):
    def __init__(self, hidden_dim: int = 200, num_layers: int = 1, bias: bool = True,
                 *, embedding_layer: nn.Embedding):
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


class LinearClassifier(nn.Module):
    def __init__(self, *, num_targets, encoder_layer: Encoder):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(
            encoder_layer.encoding_dim,
            num_targets,
        )

    def forward(self, inputs):
        return self.fc(inputs)


class Model(nn.Module):
    def __init__(self, embedding_layer: Type[Union[TokenEmbedding]],
                 encoder_layer: Type[Union[Encoder]],
                 output_layer: Type[Union[LinearClassifier]], *,
                 word_vocab: Vocab, target_vocab: Vocab = None
                 ):
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
            ignore_index=target_vocab.stoi.get('<pad>', -100),
            reduction='mean',
        )

    def forward(self, batch):
        embedding = self.embedding_layer(batch.text)
        encoding = self.encoder_layer(embedding)
        outputs = self.output_layer(encoding)
        return outputs

    def fit(self, batch):
        outputs = self(batch)
        return self.criterion(outputs, batch.label)

    def transform(self, batch):
        outputs = self(batch)
        return self.criterion(outputs, batch.label), outputs


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
def train(task: Type[Union[sst]], model: Type[Union[Model]],
          optimizer: Type[Union[sgd, adam]], num_epochs: int = 20, **kwargs):
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


app.run()
