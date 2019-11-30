import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from torchtext.data import Batch
from torchtext.vocab import Vocab

from aku import Literal, Type, Union, Tuple


class WordEmbedding(nn.Embedding):
    def __init__(self, freeze: bool = False, *, word_vocab: Vocab) -> None:
        num_embeddings, embedding_dim = word_vocab.vectors.size()
        super(WordEmbedding, self).__init__(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim,
            padding_idx=word_vocab.stoi.get('<pad>', None),
            _weight=word_vocab.vectors,
        )
        self.weight.requires_grad = not freeze

    def forward(self, x):
        if torch.is_tensor(x):
            return super(WordEmbedding, self).forward(x)
        elif not isinstance(x, PackedSequence):
            x = pack_padded_sequence(*x, batch_first=True, enforce_sorted=False)
        return x._replace(data=super(WordEmbedding, self).forward(x.data))


class LstmEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 300, bias: bool = True,
                 num_layers: int = 2, dropout: float = 0.2, *, input_dim: int):
        super(LstmEncoder, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, bias=bias,
            num_layers=num_layers, dropout=dropout,
            batch_first=True, bidirectional=True,
        )

        self.output_dim = hidden_dim

    def forward(self, embedding: PackedSequence) -> Tensor:
        _, (encoding, _) = self.rnn(embedding)
        return encoding


class ConvEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 300, bias: bool = True,
                 kernel_sizes: Tuple[int, ...] = (3, 5, 7), dropout: float = 0.2, *, input_dim: int):
        super(ConvEncoder, self).__init__()

        self.conv_layers = nn.ModuleList([
            nn.Conv2d(
                in_channels=input_dim, out_channels=input_dim, bias=bias,
                kernel_size=kernel_size, padding=kernel_size // 2,
            )
            for kernel_size in kernel_sizes
        ])
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim * len(kernel_sizes), input_dim * len(kernel_sizes)),
            nn.ReLU(),
            nn.Linear(input_dim * len(kernel_sizes), hidden_dim),
        )

        self.output_dim = hidden_dim

    def forward(self, embedding: Tensor) -> Tensor:
        encoding = torch.cat([
            layer(embedding).max(dim=1)
            for layer in self.conv_layers
        ], dim=-1)
        return self.fc(encoding)


class Classifier(nn.Sequential):
    def __init__(self, input_dim: int, bias: bool = True, *, target_vocab: Vocab):
        super(Classifier, self).__init__(
            nn.Linear(input_dim, input_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(input_dim, len(target_vocab), bias=bias),
        )


class TextClassification(nn.Module):
    def __init__(self,
                 embedding: Type[WordEmbedding],
                 encoder: Union[Type[LstmEncoder], Type[ConvEncoder]] = Type[LstmEncoder],
                 decoder: Type[Classifier] = Type[Classifier],
                 reduction: Literal['sum', 'mean'] = 'mean',
                 *,
                 word_vocab: Vocab, target_vocab: Vocab,
                 ):
        super(TextClassification, self).__init__()

        self.embedding = embedding(word_vocab=word_vocab)
        self.encoder = encoder(input_dim=self.embedding.embedding_dim)
        self.decoder = decoder(input_dim=self.encoder.output_dim, target_vocab=target_vocab)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=target_vocab.stoi.get('<pad>', -100),
            reduction=reduction,
        )

    def forward(self, batch: Batch) -> Tensor:
        if isinstance(self.encoder, LstmEncoder):
            embedding = self.embedding(batch.word)
            encoding = self.encoder(embedding)
        else:
            embedding = self.embedding(batch.word[0])
            encoding = self.encoder(embedding)
        return self.decoder(encoding)

    def fit(self, batch: Batch) -> Tensor:
        logits = self(batch)
        return self.criterion(logits, batch.target)

    def inference(self, batch: Batch) -> float:
        prediction = self(batch).argmax(dim=-1)
        return (prediction == batch.target).float().mean().item()
