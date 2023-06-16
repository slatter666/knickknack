"""
  * FileName: utils.py
  * Author:   Slatter
  * Date:     2023/6/8 14:58
  * Description:
"""
from typing import List

import torch
from tokenizers import Tokenizer, processors


class SelfTokenizer:
    def __init__(self, tokenizer: Tokenizer, max_len=128):
        self.tokenizer = tokenizer
        self.special_tokens = ['<SOS>', '<EOS>', '<PAD>']

        # initialize tokenizer
        self.tokenizer.enable_padding(pad_id=tokenizer.token_to_id('<PAD>'), pad_type_id=0, pad_token="<PAD>")
        self.tokenizer.enable_truncation(max_length=max_len)
        self.tokenizer.post_processor = processors.BertProcessing(
            ("<EOS>", self.tokenizer.token_to_id("<EOS>")),
            ("<SOS>", self.tokenizer.token_to_id("<SOS>")),
        )
        self.tokenizer.add_special_tokens(self.special_tokens)

    @classmethod
    def load_from_file(cls, file_path: str, max_len=128):
        return cls(Tokenizer.from_file(file_path))

    def get_vocab(self):
        return self.tokenizer.get_vocab()

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def encode(self, sequence):
        return self.tokenizer.encode(sequence)
        pass

    def encode_batch(self, sequences: List[str]):
        input_ids = []
        attention_mask = []
        encoding = self.tokenizer.encode_batch(sequences)
        for x in encoding:
            input_ids.append(x.ids)
            attention_mask.append(x.attention_mask)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def decode_batch(self, input):
        pass


def generate_square_subsequent_mask(sz):
    """
    Args:
        sz: sequence length
    Returns:
        square subsequent mask (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(sz, sz)).transpose(0, 1)
    return mask


def create_mask(src_padding_mask, tgt_padding_mask):
    """
    Args:
        src_padding_mask: (batch, src_len)
        tgt_padding_mask: (batch, tgt_len)

    Returns:
        src_mask: (batch, 1, src_len) or you can regard it as (batch, src_len, src_len) cause broadcasting
        tgt_mask: (batch, tgt_len, tgt_len)
    """
    tgt_len = tgt_padding_mask.size(1)

    src_mask = src_padding_mask.unsqueeze(dim=1)  # (batch, 1, src_len)

    tgt_attn_mask = generate_square_subsequent_mask(tgt_len).unsqueeze(dim=0)  # (1, tgt_len, tgt_len)
    tgt_padding_mask = tgt_padding_mask.unsqueeze(dim=1)  # (batch, 1, tgt_len)

    tgt_mask = tgt_attn_mask.logical_and(tgt_padding_mask)  # (batch, tgt_len, tgt_len)
    return src_mask, tgt_mask
