from typing import Dict

from embeddings import SentenceEmbedding
from layers import EncoderLayer
from torch import Tensor, nn
from utils import END_TOKEN, PADDING_TOKEN, START_TOKEN


class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs) -> Tensor:
        x, self_attention_mask = inputs
        for module in self._modues.values():
            y = module(x, self_attention_mask)  # 30 x 200 x 512
        return y


class Encoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_hidden: int,
        num_heads: int,
        num_layers: int,
        drop_proba: float,
        max_sequence_length: int,
        language_index: Dict[str, int],
        start_token: str,
        end_token: str,
        padding_token: str,
    ):
        super().__init__()
        self.sentence_embeddings = SentenceEmbedding(
            model_dim, max_sequence_length, language_index, drop_proba, start_token, end_token, padding_token
        )
        self.layers = SequentialEncoder(
            *[EncoderLayer(model_dim, ffn_hidden, num_heads, drop_proba) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor, self_attention_mask: Tensor) -> Tensor:
        x = self.sentence_embeddings(x)
        return self.layers(x, self_attention_mask)


if __name__ == "__main__":
    language_index = dict()
    sequence_len = 5
    max_sequence_length = 200

    dimension = 512
    num_of_heads = 8
    num_of_layers = 5
    size_ffn_hidden = 2048
    dropout_proba = 0.1
    batch = 30
    input_sequence_len = 5

    sentence_batch = None  # this should be the text input
    encoder = Encoder(
        dimension,
        size_ffn_hidden,
        num_of_heads,
        num_of_layers,
        dropout_proba,
        max_sequence_length,
        language_index,
        START_TOKEN,
        END_TOKEN,
        PADDING_TOKEN,
    )
    out = encoder(sentence_batch, self_attention_mask=None)
    print(out)
