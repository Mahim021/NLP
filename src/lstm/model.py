"""Model B -- embedding + LSTM + dense classification head."""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """text ids -> embedding -> LSTM -> dense -> 2 logits.

    The embedding is learned jointly with the LSTM from the training data
    (no pretrained vectors), as specified in PROJECT_SPEC.md Sec. 6.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128,
                 num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.3,
                 num_classes: int = 2, pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        lengths = (input_ids != self.pad_id).sum(dim=1).clamp(min=1).cpu()
        emb = self.dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            state = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            state = h_n[-1]
        return self.fc(self.dropout(state))
