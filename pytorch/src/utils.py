import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

START_TOKEN = "<START>"
END_TOKEN = "<END>"
PADDING_TOKEN = "<PADDING>"


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tuple[Tensor, Tensor]:
    """

    :param q: the query vector, which represents what I'm looking for
    :param k: the key vector, representing what I could offer
    :param v: the value vector, representing what I actually offer
    :param mask:
    :return:
    """
    # sqrt(d_k) scaling factor --> this reduces variance and put the values in a zero mean and std 1
    # for better stabilization of the learning process

    # example dimensions: q, k, v = 30x8x200x64
    d_k = q.size()[-1]  # 64
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)  # 30 x 8 x 200 x 200
    if mask:
        # to avoid the decoder to access "future" elements in the sentence
        # it also broadcast to all every batch and every head.
        # resulting in 30 x 8 x 200 x 200 masked
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)

    # get the probability values representing how much each word should focus on another in the sentence
    attention = F.softmax(scaled, dim=-1)  # 30 x 8 x 200 x 200
    values = torch.matmul(attention, v)  # 30 x 8 x 200 x 64
    return values, attention


def tokenize(
    sentence: str,
    language_index: Dict[str, int],
    max_seq_length: int = 200,
    start_token: str = START_TOKEN,
    end_token: str = END_TOKEN,
    padding_token: str = PADDING_TOKEN,
) -> Tensor:
    sentence_index = [language_index[token] for token in sentence.split(" ")]
    if start_token != "":
        sentence_index.insert(0, language_index[start_token])
    if end_token != "":
        sentence_index.append(language_index[end_token])

    padding_size = max_seq_length - len(sentence_index)
    padding_arr = [language_index[padding_token]] * padding_size
    return Tensor(sentence_index + padding_arr)


def get_default_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
