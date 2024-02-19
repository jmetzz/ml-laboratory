from typing import Dict

import torch
import torch.nn as nn
from decoder import Decoder
from encoder import Encoder
from torch import Tensor


class Transformer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_hidden: int,
        num_heads: int,
        drop_prob: float,
        num_layers: int,
        max_sequence_length: int,
        src_vocab_index: Dict[str, int],
        target_vocab_index: Dict[str, int],
        target_vocab_size: int,
        start_token: str,
        end_token: str,
        padding_token: str,
    ):
        super().__init__()
        self.encoder = Encoder(
            model_dim,
            ffn_hidden,
            num_heads,
            num_layers,
            drop_prob,
            max_sequence_length,
            src_vocab_index,
            start_token,
            end_token,
            padding_token,
        )
        self.encoder = Decoder(
            model_dim,
            ffn_hidden,
            num_heads,
            drop_prob,
            num_layers,
            max_sequence_length,
            target_vocab_index,
            start_token,
            end_token,
            padding_token,
        )
        self.liner = nn.Linear(model_dim, target_vocab_size)
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        encoder_self_attention_mask: Tensor = None,
        decoder_self_attention_mask: Tensor = None,
        decoder_cross_attention_mask: Tensor = None,
        encoder_start_token: bool = False,
        encoder_end_token: bool = False,
        decoder_start_token: bool = False,
        decoder_end_token: bool = False,
    ):
        x = self.encoder(x, encoder_self_attention_mask, start_token=encoder_start_token, end_token=encoder_end_token)
        out = self.denoder(
            x,
            y,
            decoder_self_attention_mask,
            decoder_cross_attention_mask,
            start_token=decoder_start_token,
            end_token=decoder_end_token,
        )
        return self.liner(out)


if __name__ == "__main__":
    sequence_len = 5
    max_sequence_length = 200

    model_dim = 512
    num_heads = 8
    num_layers = 5
    ffn_hidden = 2048
    drop_proba = 0.1
    batch_size = 30

    x = torch.randn((batch_size, sequence_len, model_dim))
