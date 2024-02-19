from typing import Dict

from embeddings import SentenceEmbedding
from layers import DecoderLayer
from torch import Tensor, nn


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs) -> Tensor:
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modues.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)  # 30 x 200 x 512
        return y


class Decoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_hidden: int,
        num_heads: int,
        drop_prob: float,
        num_layers: int,
        max_sequence_length: int,
        language_index: Dict[str, int],
        start_token: str,
        end_token: str,
        padding_token: str,
    ):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(
            max_sequence_length, model_dim, language_index, drop_prob, start_token, end_token, padding_token
        )
        self.layers = SequentialDecoder(
            *[DecoderLayer(model_dim, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        self_attention_mask: Tensor,
        cross_attention_mask: Tensor,
        start_token: bool,
        end_token: bool,
    ) -> Tensor:
        # x: 30 x 200 x 512
        # y: 30 x 200 x 512
        # mask: 200 x 512
        y = self.sentence_embedding(y, start_token, end_token)
        return self.layers(x, y, self_attention_mask, cross_attention_mask)  # 30 x 200 x 512
