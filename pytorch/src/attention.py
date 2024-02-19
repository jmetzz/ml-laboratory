import logging

import torch
from torch import Tensor, nn
from utils import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.d_model = model_dim  # 512
        self.num_heads = num_heads  # 8
        self.head_dimension = model_dim // num_heads  # 64
        self.qkv_layer = nn.Linear(model_dim, 3 * model_dim)  # 512 x 1536
        self.linear_layer = nn.Linear(model_dim, model_dim)  # 512 x 512

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        batch_size, sequence_length, _ = x.size()  # input: 30 x 200 x 512
        qkv = self.qkv_layer(x)  # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dimension)  # 30 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3)  # 30, 8, 200, 192
        # now break the tensor in 3 according to the last dimension. Each is 30 x 8 x 200 x 64
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product_attention(q, k, v, mask)
        # attention: 30 x 8 x 200 x 200
        # values: 30 x 8 x 200 x 64
        values = values.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.num_heads * self.head_dimension
        )  # 30 x 200 x 512
        result = self.linear_layer(values)  # output: 30 x 200 x 512
        # input and output shape matches :)

        msg = {
            "x.size()": x.size(),
            "qkv_size()": qkv.size(),
            "q size": q.size(),
            "k size": k.size(),
            "v size": v.size(),
            "attention.size()": attention.size(),
            "values.size()": values.size(),
            "result.size()": result.size(),
        }
        logging.debug("Parameters", extra=msg)

        return result


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.d_model = model_dim  # 512
        self.num_heads = num_heads  # 8
        self.head_dim = model_dim // num_heads  # 64
        self.kv_layer = nn.Linear(model_dim, 2 * model_dim)  # 512 x 1024
        self.q_layer = nn.Linear(model_dim, model_dim)  # 512 x 512
        self.linear_layer = nn.Linear(model_dim, model_dim)  # 512 x 512

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        batch_size, sequence_length, _ = x.size()  # input: 30 x 200 x 512
        kv = self.kv_layer(x)  # 30 x 200 x 1024
        q = self.q_layer(x)  # 30 x 200 x 512
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)  # 30 x 200 x 8 x 128
        kv = kv.permute(0, 2, 1, 3)  # 30 x 8 x 200 x 128
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)  # 30 x 200 x 8 x 64
        q = q.permute(0, 2, 1, 3)  # 30 x 8 x 200 x 64

        # now break the tensor in 3 according to the last dimension. Each is 30 x 8 x 200 x 64
        k, v = kv.chunk(2, dim=-1)  # k: 30 x 8 x 200 x 64  | v :30 x 8 x 200 x 64
        # We don't need the mask for cross attention. It should be None.
        values, attention = scaled_dot_product_attention(q, k, v, mask)
        # attention: 30 x 8 x 200 x 200
        # values: 30 x 8 x 200 x 64
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, dimension)  # 30 x 200 x 512
        result = self.linear_layer(values)  # output: 30 x 200 x 512
        # input and output shape matches :)

        msg = {
            "x.size()": x.size(),
            "kv_size()": kv.size(),
            "q size": q.size(),
            "k size": k.size(),
            "v size": v.size(),
            "attention.size()": attention.size(),
            "values.size()": values.size(),
            "result.size()": result.size(),
        }
        logging.debug("Parameters", extra=msg)

        return result


if __name__ == "__main__":
    sequence_len = 5
    max_sequence_length = 200

    dimension = 512
    num_of_heads = 8
    num_of_layers = 5
    size_ffn_hidden = 2048
    dropout_proba = 0.1
    batch = 30

    x = torch.randn((batch, sequence_len, dimension))

    model = MultiHeadSelfAttention(model_dim=512, num_heads=8)
    out = model.forward(x)
    print(out)

    model = MultiHeadCrossAttention(model_dim=512, num_heads=8)
    out = model.forward(x)
    print(out)
