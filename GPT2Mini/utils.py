"""
  * FileName: utils.py
  * Author:   Slatter
  * Date:     2023/6/18 21:16
  * Description:  
"""
import torch


def generate_square_subsequent_mask(sz):
    """
    Args:
        sz: sequence length
    Returns:
        square subsequent mask (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(sz, sz)).transpose(0, 1)
    return mask


def create_mask(padding_mask):
    """
    Args:
        padding_mask: (batch, seq_len)

    Returns:
        attention_mask: (batch, seq_len, seq_len)
    """
    seq_len = padding_mask.size(1)

    padding_mask = padding_mask.unsqueeze(dim=1)  # (batch, 1, tgt_len)
    attention_mask = generate_square_subsequent_mask(seq_len).unsqueeze(dim=0)  # (1, seq_len, seq_len)

    attention_mask = attention_mask.logical_and(padding_mask)  # (batch, tgt_len, tgt_len)
    return attention_mask
