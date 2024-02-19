from typing import List

import torch
from attention import MultiHeadCrossAttention, MultiHeadSelfAttention
from torch import Tensor, nn


class NormalizationLayer(nn.Module):
    def __init__(self, parameters_shape: List, eps: int = 1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape  # [512]
        self.eps = eps  # to avoid division by zero
        # learnable parameters for this layer
        self.gamma = nn.Parameter(torch.ones(parameters_shape))  # 512
        self.beta = nn.Parameter(torch.zeros(parameters_shape))  # 512

    def forward(self, inputs: Tensor) -> Tensor:  # 30 x 200 x 512
        dimensions = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dimensions, keepdim=True)  # 30 x 200 x 1
        var = ((inputs - mean) ** 2).mean(dim=dimensions, keepdims=True)  # 30 x 200 x 1
        std = (var + self.eps).sqrt()  # 30 x 200 x 1
        y = (inputs - mean) / std  # 30 x 200 x 512
        return self.gamma * y + self.beta  # 30 x 200 x 512


class PositionAwareFeedForwardLayer(nn.Module):
    def __init__(self, model_dim: int, num_hidden: int, drop_proba: float):
        super().__init__()
        self.linear1 = nn.Linear(model_dim, num_hidden)  # 512 x 2048
        self.relu = nn.ReLU()  # 512 x 2048
        self.dropout = nn.Dropout(drop_proba)  # 512 x 2048
        self.linear2 = nn.Linear(num_hidden, model_dim)  # 2048 x 512

    def forward(self, x: Tensor) -> Tensor:  # 30 x 200 x 512
        x = self.linear1(x)  # 30 x 200 x 2048
        x = self.relu(x)  # 30 x 200 x 2048
        x = self.dropout(x)  # 30 x 200 x 2048
        return self.linear2(x)  # 30 x 200 x 512


class EncoderLayer(nn.Module):
    def __init__(self, model_dim: int, ffn_hidden: int, num_heads: int, drop_proba: float):
        super().__init__()
        self.attention = MultiHeadSelfAttention(model_dim, num_heads)
        self.norm1 = NormalizationLayer(parameters_shape=[model_dim])
        self.dropout1 = nn.Dropout(drop_proba)

        self.ffn = PositionAwareFeedForwardLayer(model_dim, ffn_hidden, drop_proba)
        self.norm2 = NormalizationLayer(parameters_shape=[model_dim])
        self.dropout2 = nn.Dropout(p=drop_proba)

    def forward(self, x: Tensor, self_attention_mask: Tensor) -> Tensor:
        residual_x = x.clone()  # 30 x 200 x 512
        x = self.attention(x, self_attention_mask)  # 30 x 200 x 512
        x = self.dropout1(x)  # 30 x 200 x 512
        x = self.norm1(x + residual_x)  # 30 x 200 x 512

        # residual_x = x  # 30 x 200 x 512
        x = self.ffn(x)  # 30 x 200 x 512
        x = self.dropout2(x)  # 30 x 200 x 512
        return self.norm2(x + residual_x)  # 30 x 200 x 512


class DecoderLayer(nn.Module):
    def __init__(self, model_dim: int, ffn_hidden: int, num_heads: int, drop_proba: float):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(model_dim, num_heads)
        self.norm1 = NormalizationLayer(parameters_shape=[model_dim])
        self.dropout1 = nn.Dropout(drop_proba)

        self.encoder_decoder_attention = MultiHeadCrossAttention(model_dim, num_heads)
        self.norm2 = NormalizationLayer(parameters_shape=[model_dim])
        self.dropout2 = nn.Dropout(drop_proba)

        self.ffn = PositionAwareFeedForwardLayer(model_dim, ffn_hidden, drop_proba)
        self.norm3 = NormalizationLayer(parameters_shape=[model_dim])
        self.dropout3 = nn.Dropout(drop_proba)

    def forward(self, x, y, decoder_mask):
        _y = y.clone()  # 30 x 200 x 512
        y = self.self_attention(y, mask=decoder_mask)
        y = self.dropout1(y)
        y = self.norm1(y + _y)

        # _y = y
        # apply cross-attention:
        y = self.encoder_decoder_attention(x, y, mask=None)  # 30 x 200 x 512
        y = self.dropout2(y)  # 30 x 200 x 512
        y = self.norm2(y + _y)  # 30 x 200 x 512

        # _y = y
        y = self.ffn(y)  # 30 x 200 x 512
        y = self.dropout3(y)  # 30 x 200 x 512
        return self.norm3(y + _y)  # 30 x 200 x 512
