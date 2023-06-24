"""
  * FileName: arch.py
  * Author:   Slatter
  * Date:     2023/6/17 15:49
  * Description:
"""
import copy
import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn.init import constant_, normal_


class Attention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            embed_dim: Total dimension of the model.
            num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
                across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
            dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        """
        super(Attention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_(m.weight, 0, 0.02)
                constant_(m.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor, need_weights: bool = True):
        """
        Args:
            query: Query embeddings of shape `(N, L, E_q)` for batched input, where `N` is the batch size,
                `L` is the maximum sequence length, and `E_q` is the query embedding dimension `embed_dim`.
                Queries are compared against key-value pairs to produce the output.
            key: Key embeddings of shape `(N, L, E_k)` for batched input, where `N` is the batch size,
                `L` is the maximum sequence length, and `E_k` is the query embedding dimension `embed_dim`.
            value: Value embeddings of shape `(N, L, E_v)` for batched input, where `N` is the batch size,
                `L` is the maximum sequence length, and `E_v` is the query embedding dimension `embed_dim`.
            attn_mask: A mask of shape `(N, L, L)`  indicating which elements within `key` to ignore for the
                purpose of attention (i.e. treat as "padding").
                Binary and byte masks are supported.
                For a binary mask, a `False` value indicates that the corresponding `key` value will be ignored
                for the purpose of attention. For a byte mask, a zero value indicates that the corresponding `key`
                value will be ignored.
            need_weights:  If specified, returns `attn_output_weights` in addition to `attn_outputs`. Default: `True`.
        Returns:
            - attn_output: Attention outputs of shape `(N, L, E)` where `L` is the maximum sequence length,
                `N` is the batch size, and `E` is the embedding dimension ``embed_dim``.
            - attn_weight: Attention weight of shape `(N, num_heads, L, L)`
        """
        bsz = query.size(0)
        # perform linear operation and split into num_heads
        # (N, L, embed_size) -> (N, L, num_heads, head_dim)
        query = self.q_proj(query).view(bsz, -1, self.num_heads, self.head_dim)
        key = self.k_proj(key).view(bsz, -1, self.num_heads, self.head_dim)
        value = self.v_proj(value).view(bsz, -1, self.num_heads, self.head_dim)

        # (N, L, num_heads, head_dim) -> (N, num_heads, L, head_dim)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # calculate attention weights
        # (N, num_heads, L, head_dim) * (N, num_heads, head_dim, L) -> (N, num_heads, L, L)
        attn_weights = torch.einsum('ijkl,ijml->ijkm', query, key) / math.sqrt(self.head_dim)

        attn_mask = attn_mask.unsqueeze(dim=1)  # (N, 1, L, L)

        # apply attention mask
        attn_weights.masked_fill_(attn_mask == 0, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_outputs = self.dropout(attn_weights)
        # (N, num_heads, L, L) * (N, num_heads, L, head_dim) -> (N, num_heads, L, head_dim)
        attn_outputs = torch.matmul(attn_outputs, value)

        attn_outputs = attn_outputs.transpose(1, 2).contiguous().view(bsz, -1, self.embed_dim)  # (N, L, embed_dim)
        attn_outputs = self.out_proj(attn_outputs)
        if need_weights:
            return attn_outputs, attn_weights
        else:
            return attn_outputs, None


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = 'relu', norm_first: bool = True):
        """
        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multi-head attention models (required).
            dim_feedforward: the dimension of the feedforward network model (default=2048).
            dropout: the dropout value (default=0.1).
            activation: the activation function of the intermediate layer, a string ("relu" or "gelu")
            norm_first: if `True`, layer norm is done prior to attention and feedforward operations, respectively.
                Otherwise, it's done after. Default: `False` (after).
        """
        super(TransformerBlock, self).__init__()
        self.norm_first = norm_first
        self.self_attn = Attention(d_model, nhead, dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise NotImplementedError

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                normal_(m.weight, 0, 0.02)
                constant_(m.bias, 0.)

    def forward(self, src: Tensor, attn_mask: Tensor, need_weights: bool = False):
        """
        Args:
            src: the sequence to the transformer block (required) of shape (N, L, E) where `N` is the batch size,
                `L` is the maximum sequence length, `E` is the embedding size,
            attn_mask: the mask for the sequence (required) of shape (N, L, L)
            need_weights: `attn_output_weights` in addition to `attn_outputs`. Default: `False`.
        Returns:
            - outputs: Transformer Block outputs of shape `(N, L, E)` where `L` is the maximum sequence length,
                `N` is the batch size, and `E` is the embedding dimension `embed_dim`.
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask, need_weights)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask, need_weights))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], need_weights) -> Tensor:
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=need_weights)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = 'gelu', norm_first: bool = True):
        """
        Args:
            num_layers: the number of transformer layers (required).
            check other parameters in TransformerBlock
        """
        super(Transformer, self).__init__()
        layer = TransformerBlock(d_model, nhead, dim_feedforward, dropout, activation, norm_first)
        self.layers = _get_clones(layer, num_layers)

    def forward(self, src: Tensor, attn_mask: Optional[Tensor] = None, need_weights: bool = False):
        x = src
        for mod in self.layers:
            x = mod(x, attn_mask=attn_mask, need_weights=need_weights)

        return x

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> Tensor:
        """
        Generate a square mask for the sequence. The masked positions are filled with False. Unmasked positions are
        filled with True.
        Args:
            sz: sequence length
        Returns:
            mask: mask of shape (sz, sz)
        """
        return torch.triu(torch.ones((sz, sz))).transpose(0, 1)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
