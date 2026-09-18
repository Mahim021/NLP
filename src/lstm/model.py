"""LSTM classifier using pretrained word embeddings."""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """
    Token IDs -> pretrained embedding -> BiLSTM -> dense -> Yes/No.
    """

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        num_classes: int = 2,
        pad_id: int = 0,
        embedding_trainable: bool = True,
    ):
        super().__init__()

        self.pad_id = pad_id

        vocab_size, embed_dim = embedding_matrix.shape

        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=not embedding_trainable,
            padding_idx=pad_id,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        output_dim = hidden_dim * (2 if bidirectional else 1)

        self.fc = nn.Linear(
            output_dim,
            num_classes,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:

        lengths = (
            (input_ids != self.pad_id)
            .sum(dim=1)
            .clamp(min=1)
            .cpu()
        )

        embeddings = self.dropout(
            self.embedding(input_ids)
        )

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        if self.lstm.bidirectional:
            state = torch.cat(
                [hidden[-2], hidden[-1]],
                dim=1,
            )
        else:
            state = hidden[-1]

        state = self.dropout(state)

        return self.fc(state)