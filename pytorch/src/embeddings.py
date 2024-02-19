from typing import Dict, List

import torch
from torch import Tensor, nn
from utils import END_TOKEN, PADDING_TOKEN, START_TOKEN, get_default_device, tokenize


class SentenceEmbedding(nn.Module):
    """Generate the embedding representing a given sentence"""

    def __init__(
        self,
        model_dim: int,
        max_sequence_length: int,
        language_index: Dict[str, int],
        drop_proba: float = 0.1,
        start_token: str = START_TOKEN,
        end_token: str = END_TOKEN,
        padding_token: str = PADDING_TOKEN,
    ):
        super().__init__()
        self.vocab_size = len(language_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, model_dim)
        self.language_index = language_index
        self.position_encoder = PositionalEncodingLayer(model_dim, max_sequence_length)
        self.dropout = nn.Dropout(drop_proba)
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token

    def forward(self, x: Tensor):
        x = self.batch_tokenize(x)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_default_device())
        return self.dropout(x + pos)

    def batch_tokenize(self, batch: List[str]) -> Tensor:
        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append(
                tokenize(
                    batch[sentence_num],
                    self.language_index,
                    self.max_sequence_length,
                    self.start_token,
                    self.end_token,
                    self.padding_token,
                )
            )
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_default_device())


class PositionalEncodingLayer(nn.Module):
    def __init__(self, model_dim: int, max_sequence_length: int):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.model_dim = model_dim

    def forward(self) -> Tensor:
        even_i = torch.arange(0, self.model_dim, 2).float()
        denominator = torch.pow(10000, even_i / self.model_dim)
        position = torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1)
        even_position_encoding = torch.sin(position / denominator)
        odd_position_encoding = torch.cos(position / denominator)
        stacked = torch.stack([even_position_encoding, odd_position_encoding], dim=2)
        return torch.flatten(stacked, start_dim=1, end_dim=2)
