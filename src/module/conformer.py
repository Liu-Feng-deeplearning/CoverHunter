#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2022/8/30 11:59 AM
# software: PyCharm

""" Conformer Structure from Wenet(Espnet) """

from __future__ import division, absolute_import

import logging
import math
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from typeguard import check_argument_types

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# # *** Mask and Activation *** # #
def subsequent_chunk_mask(
    size: int,
    chunk_size: int,
    num_left_chunks: int = -1,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
  """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
  ret = torch.zeros(size, size, device=device, dtype=torch.bool)
  for i in range(size):
    if num_left_chunks < 0:
      start = 0
    else:
      start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
    ending = min((i // chunk_size + 1) * chunk_size, size)
    ret[i, start:ending] = True
  return ret


def add_optional_chunk_mask(xs: torch.Tensor, masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int, static_chunk_size: int,
                            num_decoding_left_chunks: int):
  """ Apply optional mask for encoder.

    Args:
        masks:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
  # Whether to use chunk mask or not
  if use_dynamic_chunk:
    max_len = xs.size(1)
    if decoding_chunk_size < 0:
      chunk_size = max_len
      num_left_chunks = -1
    elif decoding_chunk_size > 0:
      chunk_size = decoding_chunk_size
      num_left_chunks = num_decoding_left_chunks
    else:
      # chunk size is either [1, 25] or full context(max_len).
      # Since we use 4 times subsampling and allow up to 1s(100 frames)
      # delay, the maximum frame is 100 / 4 = 25.
      chunk_size = torch.randint(1, max_len, (1,)).item()
      num_left_chunks = -1
      if chunk_size > max_len // 2:
        chunk_size = max_len
      else:
        chunk_size = chunk_size % 25 + 1
        if use_dynamic_left_chunk:
          max_left_chunks = (max_len - 1) // chunk_size
          num_left_chunks = torch.randint(0, max_left_chunks,
                                          (1,)).item()
    chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size,
                                        num_left_chunks,
                                        xs.device)  # (L, L)
    chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
    chunk_masks = masks & chunk_masks  # (B, L, L)
  elif static_chunk_size > 0:
    num_left_chunks = num_decoding_left_chunks
    chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size,
                                        num_left_chunks,
                                        xs.device)  # (L, L)
    chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
    chunk_masks = masks & chunk_masks  # (B, L, L)
  else:
    chunk_masks = masks
  return chunk_masks


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
  """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
        max_len:

    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> make_pad_mask(torch.tensor([5, 3, 2]))
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
  batch_size = lengths.size(0)
  max_len = max_len if max_len > 0 else lengths.max().item()
  seq_range = torch.arange(0,
                           max_len,
                           dtype=torch.int64,
                           device=lengths.device)
  seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
  seq_length_expand = lengths.unsqueeze(-1)
  mask = seq_range_expand >= seq_length_expand
  return mask


class Swish(torch.nn.Module):
  """Construct an Swish object."""

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Return Swish activation function."""
    return x * torch.sigmoid(x)


def get_activation(act):
  """Return activation function."""
  activation_funcs = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU
  }
  return activation_funcs[act]()


# # *** Position Encoding *** # #
class PositionalEncoding(torch.nn.Module):
  """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

  def __init__(self,
               d_model: int,
               dropout_rate: float,
               max_len: int = 5000,
               reverse: bool = False):
    """Construct an PositionalEncoding object."""
    super().__init__()
    self.d_model = d_model
    self.xscale = math.sqrt(self.d_model)
    self.dropout = torch.nn.Dropout(p=dropout_rate)
    self.max_len = max_len

    self.pe = torch.zeros(self.max_len, self.d_model)
    position = torch.arange(0, self.max_len,
                            dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0, self.d_model, 2, dtype=torch.float32) *
      -(math.log(10000.0) / self.d_model))
    self.pe[:, 0::2] = torch.sin(position * div_term)
    self.pe[:, 1::2] = torch.cos(position * div_term)
    self.pe = self.pe.unsqueeze(0)

  def forward(self,
              x: torch.Tensor,
              offset: Union[int, torch.Tensor] = 0) \
      -> Tuple[torch.Tensor, torch.Tensor]:
    """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int, torch.tensor): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """

    self.pe = self.pe.to(x.device)
    pos_emb = self.position_encoding(offset, x.size(1), False)
    x = x * self.xscale + pos_emb
    return self.dropout(x), self.dropout(pos_emb)

  def position_encoding(self, offset: Union[int, torch.Tensor], size: int,
                        apply_dropout: bool = True) -> torch.Tensor:
    """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            apply_dropout:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
    # How to subscript a Union type:
    #   https://github.com/pytorch/pytorch/issues/69434
    if isinstance(offset, int):
      assert offset + size < self.max_len
      pos_emb = self.pe[:, offset:offset + size]
    elif isinstance(offset, torch.Tensor) and offset.dim() == 0:  # scalar
      assert offset + size < self.max_len
      pos_emb = self.pe[:, offset:offset + size]
    else:  # for batched streaming decoding on GPU
      assert torch.max(offset) + size < self.max_len
      index = offset.unsqueeze(1) + torch.arange(0, size).to(offset.device)  # B X T
      flag = index > 0
      # remove negative offset
      index = index * flag
      pos_emb = F.embedding(index, self.pe[0])  # B X T X d_model

    if apply_dropout:
      pos_emb = self.dropout(pos_emb)
    return pos_emb


class RelPositionalEncoding(PositionalEncoding):
  """Relative positional encoding module.
    See : Appendix B in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

  def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
    """Initialize class."""
    super().__init__(d_model, dropout_rate, max_len, reverse=True)

  def forward(self,
              x: torch.Tensor,
              offset: Union[int, torch.Tensor] = 0) \
      -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Positional embedding tensor (1, time, `*`).
        """
    self.pe = self.pe.to(x.device)
    x = x * self.xscale
    pos_emb = self.position_encoding(offset, x.size(1), False)
    return self.dropout(x), self.dropout(pos_emb)


class NoPositionalEncoding(torch.nn.Module):
  """ No position encoding
    """

  def __init__(self, d_model: int, dropout_rate: float):
    super().__init__()
    self.d_model = d_model
    self.dropout = torch.nn.Dropout(p=dropout_rate)

  def forward(self,
              x: torch.Tensor,
              offset: Union[int, torch.Tensor] = 0) \
      -> Tuple[torch.Tensor, torch.Tensor]:
    """ Just return zero vector for interface compatibility
        """
    pos_emb = torch.zeros(1, x.size(1), self.d_model).to(x.device)
    return self.dropout(x), pos_emb

  def position_encoding(
      self, offset: Union[int, torch.Tensor], size: int) -> torch.Tensor:
    return torch.zeros(1, size, self.d_model)


# # *** Subsample *** # #
class BaseSubsampling(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.right_context = 0
    self.subsampling_rate = 1

  def position_encoding(self, offset: Union[int, torch.Tensor],
                        size: int) -> torch.Tensor:
    return self.pos_enc.position_encoding(offset, size)


class LinearNoSubsampling(BaseSubsampling):
  """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

  def __init__(self, idim: int, odim: int, dropout_rate: float,
               pos_enc_class: torch.nn.Module):
    """Construct an linear object."""
    super().__init__()
    self.out = torch.nn.Sequential(
      torch.nn.Linear(idim, odim),
      torch.nn.LayerNorm(odim, eps=1e-5),
      torch.nn.Dropout(dropout_rate),
    )
    self.pos_enc = pos_enc_class
    self.right_context = 0
    self.subsampling_rate = 1

  def forward(
      self,
      x: torch.Tensor,
      x_mask: torch.Tensor,
      offset: Union[int, torch.Tensor] = 0
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
    x = self.out(x)
    x, pos_emb = self.pos_enc(x, offset)
    return x, pos_emb, x_mask


class Conv2dSubsampling4(BaseSubsampling):
  """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

  def __init__(self, idim: int, odim: int, dropout_rate: float,
               pos_enc_class: torch.nn.Module):
    """Construct an Conv2dSubsampling4 object."""
    super().__init__()
    self.conv = torch.nn.Sequential(
      torch.nn.Conv2d(1, odim, 3, 2),
      torch.nn.ReLU(),
      torch.nn.Conv2d(odim, odim, 3, 2),
      torch.nn.ReLU(),
    )
    self.out = torch.nn.Sequential(
      torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
    self.pos_enc = pos_enc_class
    # The right context for every conv layer is computed by:
    # (kernel_size - 1) * frame_rate_of_this_layer
    self.subsampling_rate = 4
    # 6 = (3 - 1) * 1 + (3 - 1) * 2
    self.right_context = 6

  def forward(
      self,
      x: torch.Tensor,
      x_mask: torch.Tensor,
      offset: Union[int, torch.Tensor] = 0
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
    x = x.unsqueeze(1)  # (b, c=1, t, f)
    x = self.conv(x)
    b, c, t, f = x.size()
    x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
    x, pos_emb = self.pos_enc(x, offset)
    return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2]


class Conv2dSubsampling8(BaseSubsampling):
  """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

  def __init__(self, idim: int, odim: int, dropout_rate: float,
               pos_enc_class: torch.nn.Module):
    """Construct an Conv2dSubsampling8 object."""
    super().__init__()
    self.conv = torch.nn.Sequential(
      torch.nn.Conv2d(1, odim, 3, 2),
      torch.nn.ReLU(),
      torch.nn.Conv2d(odim, odim, 3, 2),
      torch.nn.ReLU(),
      torch.nn.Conv2d(odim, odim, 3, 2),
      torch.nn.ReLU(),
    )
    self.linear = torch.nn.Linear(
      odim * ((((idim - 1) // 2 - 1) // 2 - 1) // 2), odim)
    self.pos_enc = pos_enc_class
    self.subsampling_rate = 8
    # 14 = (3 - 1) * 1 + (3 - 1) * 2 + (3 - 1) * 4
    self.right_context = 14

  def forward(
      self,
      x: torch.Tensor,
      x_mask: torch.Tensor,
      offset: Union[int, torch.Tensor] = 0
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.
            torch.Tensor: positional encoding
        """
    x = x.unsqueeze(1)  # (b, c, t, f)
    x = self.conv(x)
    b, c, t, f = x.size()
    x = self.linear(x.transpose(1, 2).contiguous().view(b, t, c * f))
    x, pos_emb = self.pos_enc(x, offset)
    return x, pos_emb, x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]


# # *** Attention *** # #
class MultiHeadedAttention(nn.Module):
  """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

  def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
    """Construct an MultiHeadedAttention object."""
    super().__init__()
    assert n_feat % n_head == 0
    # We assume d_v always equals d_k
    self.d_k = n_feat // n_head
    self.h = n_head
    self.linear_q = nn.Linear(n_feat, n_feat)
    self.linear_k = nn.Linear(n_feat, n_feat)
    self.linear_v = nn.Linear(n_feat, n_feat)
    self.linear_out = nn.Linear(n_feat, n_feat)
    self.dropout = nn.Dropout(p=dropout_rate)

  def forward_qkv(
      self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor, size
                (#batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor, size
                (#batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor, size
                (#batch, n_head, time2, d_k).

        """
    n_batch = query.size(0)
    q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
    k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
    v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
    q = q.transpose(1, 2)  # (batch, head, time1, d_k)
    k = k.transpose(1, 2)  # (batch, head, time2, d_k)
    v = v.transpose(1, 2)  # (batch, head, time2, d_k)

    return q, k, v

  def forward_attention(
      self, value: torch.Tensor, scores: torch.Tensor,
      mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
  ) -> torch.Tensor:
    """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value, size
                (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score, size
                (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
    n_batch = value.size(0)
    # NOTE(xcsong): When will `if mask.size(2) > 0` be True?
    #   1. onnx(16/4) [WHY? Because we feed real cache & real mask for the
    #           1st chunk to ease the onnx export.]
    #   2. pytorch training
    if mask.size(2) > 0:  # time2 > 0
      mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
      # For last chunk, time2 might be larger than scores.size(-1)
      mask = mask[:, :, :, :scores.size(-1)]  # (batch, 1, *, time2)
      scores = scores.masked_fill(mask, -float('inf'))
      attn = torch.softmax(scores, dim=-1).masked_fill(
        mask, 0.0)  # (batch, head, time1, time2)
    # NOTE(xcsong): When will `if mask.size(2) > 0` be False?
    #   1. onnx(16/-1, -1/-1, 16/0)
    #   2. jit (16/-1, -1/-1, 16/0, 16/4)
    else:
      attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

    p_attn = self.dropout(attn)
    x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
    x = (x.transpose(1, 2).contiguous().view(n_batch, -1,
                                             self.h * self.d_k)
         )  # (batch, time1, d_model)

    return self.linear_out(x)  # (batch, time1, d_model)

  def forward(self, query: torch.Tensor, key: torch.Tensor,
              value: torch.Tensor,
              mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
              pos_emb: torch.Tensor = torch.empty(0),
              cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).
                1.When applying cross attention between decoder and encoder,
                the batch padding mask for input is in (#batch, 1, T) shape.
                2.When applying self attention of encoder,
                the mask is in (#batch, T, T)  shape.
                3.When applying self attention of decoder,
                the mask is in (#batch, L, L)  shape.
                4.If the different position in decoder see different block
                of the encoder, such as Mocha, the passed in mask could be
                in (#batch, L, T) shape. But there is no such case in current
                Wenet.
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`


        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`

        """
    q, k, v = self.forward_qkv(query, key, value)

    # NOTE(xcsong):
    #   when export onnx model, for 1st chunk, we feed
    #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
    #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
    #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
    #       and we will always do splitting and
    #       concatnation(this will simplify onnx export). Note that
    #       it's OK to concat & split zero-shaped tensors(see code below).
    #   when export jit  model, for 1st chunk, we always feed
    #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
    # >>> a = torch.ones((1, 2, 0, 4))
    # >>> b = torch.ones((1, 2, 3, 4))
    # >>> c = torch.cat((a, b), dim=2)
    # >>> torch.equal(b, c)        # True
    # >>> d = torch.split(a, 2, dim=-1)
    # >>> torch.equal(d[0], d[1])  # True
    if cache.size(0) > 0:
      key_cache, value_cache = torch.split(
        cache, cache.size(-1) // 2, dim=-1)
      k = torch.cat([key_cache, k], dim=2)
      v = torch.cat([value_cache, v], dim=2)
    # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
    #   non-trivial to calculate `next_cache_start` here.
    new_cache = torch.cat((k, v), dim=-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    return self.forward_attention(v, scores, mask), new_cache


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
  """Multi-Head Attention layer with relative position encoding.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

  def __init__(self, n_head, n_feat, dropout_rate):
    """Construct an RelPositionMultiHeadedAttention object."""
    super().__init__(n_head, n_feat, dropout_rate)
    # linear transformation for positional encoding
    self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
    # these two learnable bias are used in matrix c and matrix d
    # as described in https://arxiv.org/abs/1901.02860 Section 3.3
    self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
    self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
    torch.nn.init.xavier_uniform_(self.pos_bias_u)
    torch.nn.init.xavier_uniform_(self.pos_bias_v)

  def rel_shift(self, x, zero_triu: bool = False):
    """Compute relative positinal encoding.
        Args:
            x (torch.Tensor): Input tensor (batch, time, size).
            zero_triu (bool): If true, return the lower triangular part of
                the matrix.
        Returns:
            torch.Tensor: Output tensor.
        """

    zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1),
                           device=x.device,
                           dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=-1)

    x_padded = x_padded.view(x.size()[0],
                             x.size()[1],
                             x.size(3) + 1, x.size(2))
    x = x_padded[:, :, 1:].view_as(x)

    if zero_triu:
      ones = torch.ones((x.size(2), x.size(3)))
      x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

    return x

  def forward(self, query: torch.Tensor,
              key: torch.Tensor, value: torch.Tensor,
              mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
              pos_emb: torch.Tensor = torch.empty(0),
              cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2), (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): Positional embedding tensor
                (#batch, time2, size).
            cache (torch.Tensor): Cache tensor (1, head, cache_t, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).
            torch.Tensor: Cache tensor (1, head, cache_t + time1, d_k * 2)
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
    q, k, v = self.forward_qkv(query, key, value)
    q = q.transpose(1, 2)  # (batch, time1, head, d_k)

    # NOTE(xcsong):
    #   when export onnx model, for 1st chunk, we feed
    #       cache(1, head, 0, d_k * 2) (16/-1, -1/-1, 16/0 mode)
    #       or cache(1, head, real_cache_t, d_k * 2) (16/4 mode).
    #       In all modes, `if cache.size(0) > 0` will alwayse be `True`
    #       and we will always do splitting and
    #       concatnation(this will simplify onnx export). Note that
    #       it's OK to concat & split zero-shaped tensors(see code below).
    #   when export jit  model, for 1st chunk, we always feed
    #       cache(0, 0, 0, 0) since jit supports dynamic if-branch.
    # >>> a = torch.ones((1, 2, 0, 4))
    # >>> b = torch.ones((1, 2, 3, 4))
    # >>> c = torch.cat((a, b), dim=2)
    # >>> torch.equal(b, c)        # True
    # >>> d = torch.split(a, 2, dim=-1)
    # >>> torch.equal(d[0], d[1])  # True
    if cache.size(0) > 0:
      key_cache, value_cache = torch.split(
        cache, cache.size(-1) // 2, dim=-1)
      k = torch.cat([key_cache, k], dim=2)
      v = torch.cat([value_cache, v], dim=2)
    # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
    #   non-trivial to calculate `next_cache_start` here.
    new_cache = torch.cat((k, v), dim=-1)

    n_batch_pos = pos_emb.size(0)
    p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
    p = p.transpose(1, 2)  # (batch, head, time1, d_k)

    # (batch, head, time1, d_k)
    q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
    # (batch, head, time1, d_k)
    q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

    # compute attention score
    # first compute matrix a and matrix c
    # as described in https://arxiv.org/abs/1901.02860 Section 3.3
    # (batch, head, time1, time2)
    matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

    # compute matrix b and matrix d
    # (batch, head, time1, time2)
    matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
    # Remove rel_shift since it is useless in speech recognition,
    # and it requires special attention for streaming.
    # matrix_bd = self.rel_shift(matrix_bd)

    scores = (matrix_ac + matrix_bd) / math.sqrt(
      self.d_k)  # (batch, head, time1, time2)

    return self.forward_attention(v, scores, mask), new_cache


# # *** Conv Module *** # #
class ConvolutionModule(nn.Module):
  """ConvolutionModule in Conformer model."""

  def __init__(self,
               channels: int,
               kernel_size: int = 15,
               activation: nn.Module = nn.ReLU(),
               norm: str = "batch_norm",
               causal: bool = False,
               bias: bool = True):
    """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
    assert check_argument_types()
    super().__init__()

    self.pointwise_conv1 = nn.Conv1d(
      channels,
      2 * channels,
      kernel_size=1,
      stride=1,
      padding=0,
      bias=bias,
    )
    # self.lorder is used to distinguish if it's a causal convolution,
    # if self.lorder > 0: it's a causal convolution, the input will be
    #    padded with self.lorder frames on the left in forward.
    # else: it's a symmetrical convolution
    if causal:
      padding = 0
      self.lorder = kernel_size - 1
    else:
      # kernel_size should be an odd number for none causal convolution
      assert (kernel_size - 1) % 2 == 0
      padding = (kernel_size - 1) // 2
      self.lorder = 0
    self.depthwise_conv = nn.Conv1d(
      channels,
      channels,
      kernel_size,
      stride=1,
      padding=padding,
      groups=channels,
      bias=bias,
    )

    assert norm in ['batch_norm', 'layer_norm']
    if norm == "batch_norm":
      self.use_layer_norm = False
      self.norm = nn.BatchNorm1d(channels)
    else:
      self.use_layer_norm = True
      self.norm = nn.LayerNorm(channels)

    self.pointwise_conv2 = nn.Conv1d(
      channels,
      channels,
      kernel_size=1,
      stride=1,
      padding=0,
      bias=bias,
    )
    self.activation = activation

  def forward(
      self,
      x: torch.Tensor,
      mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
      cache: torch.Tensor = torch.zeros((0, 0, 0)),
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
    # exchange the temporal dimension and the feature dimension
    x = x.transpose(1, 2)  # (#batch, channels, time)

    # mask batch padding
    if mask_pad.size(2) > 0:  # time > 0
      x.masked_fill_(~mask_pad, 0.0)

    if self.lorder > 0:
      if cache.size(2) == 0:  # cache_t == 0
        x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
      else:
        assert cache.size(0) == x.size(0)  # equal batch
        assert cache.size(1) == x.size(1)  # equal channel
        x = torch.cat((cache, x), dim=2)
      assert (x.size(2) > self.lorder)
      new_cache = x[:, :, -self.lorder:]
    else:
      # It's better we just return None if no cache is required,
      # However, for JIT export, here we just fake one tensor instead of
      # None.
      new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

    # GLU mechanism
    x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
    x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

    # 1D Depthwise Conv
    x = self.depthwise_conv(x)
    if self.use_layer_norm:
      x = x.transpose(1, 2)
    x = self.activation(self.norm(x))
    if self.use_layer_norm:
      x = x.transpose(1, 2)
    x = self.pointwise_conv2(x)
    # mask batch padding
    if mask_pad.size(2) > 0:  # time > 0
      x.masked_fill_(~mask_pad, 0.0)

    return x.transpose(1, 2), new_cache


# # *** FeedForward Module *** # #
class PositionwiseFeedForward(torch.nn.Module):
  """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

  def __init__(self,
               idim: int,
               hidden_units: int,
               dropout_rate: float,
               activation: torch.nn.Module = torch.nn.ReLU()):
    """Construct a PositionwiseFeedForward object."""
    super(PositionwiseFeedForward, self).__init__()
    self.w_1 = torch.nn.Linear(idim, hidden_units)
    self.activation = activation
    self.dropout = torch.nn.Dropout(dropout_rate)
    self.w_2 = torch.nn.Linear(hidden_units, idim)

  def forward(self, xs: torch.Tensor) -> torch.Tensor:
    """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
    return self.w_2(self.dropout(self.activation(self.w_1(xs))))


# # *** Conformer *** # #
class ConformerEncoderLayer(torch.nn.Module):
  """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
        concat_after (bool): Whether to concat attention layer's input and
            output.
            True: x -> x + linear(concat(x, att(x)))
            False: x -> x + att(x)
    """

  def __init__(
      self,
      size: int,
      self_attn: torch.nn.Module,
      feed_forward: Optional[nn.Module] = None,
      feed_forward_macaron: Optional[nn.Module] = None,
      conv_module: Optional[nn.Module] = None,
      dropout_rate: float = 0.1,
      normalize_before: bool = True,
      concat_after: bool = False,
  ):
    """Construct an EncoderLayer object."""
    super().__init__()
    self.self_attn = self_attn
    self.feed_forward = feed_forward
    self.feed_forward_macaron = feed_forward_macaron
    self.conv_module = conv_module
    self.norm_ff = nn.LayerNorm(size, eps=1e-5)  # for the FNN module
    self.norm_mha = nn.LayerNorm(size, eps=1e-5)  # for the MHA module
    if feed_forward_macaron is not None:
      self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
      self.ff_scale = 0.5
    else:
      self.ff_scale = 1.0
    if self.conv_module is not None:
      self.norm_conv = nn.LayerNorm(size,
                                    eps=1e-5)  # for the CNN module
      self.norm_final = nn.LayerNorm(
        size, eps=1e-5)  # for the final output of the block
    self.dropout = nn.Dropout(dropout_rate)
    self.size = size
    self.normalize_before = normalize_before
    self.concat_after = concat_after
    if self.concat_after:
      self.concat_linear = nn.Linear(size + size, size)
    else:
      self.concat_linear = nn.Identity()

  def forward(
      self,
      x: torch.Tensor,
      mask: torch.Tensor,
      pos_emb: torch.Tensor,
      mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
      att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
      cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

    # whether to use macaron style
    if self.feed_forward_macaron is not None:
      residual = x
      if self.normalize_before:
        x = self.norm_ff_macaron(x)
      x = residual + self.ff_scale * self.dropout(
        self.feed_forward_macaron(x))
      if not self.normalize_before:
        x = self.norm_ff_macaron(x)

    # multi-headed self-attention module
    residual = x
    if self.normalize_before:
      x = self.norm_mha(x)

    x_att, new_att_cache = self.self_attn(
      x, x, x, mask, pos_emb, att_cache)
    if self.concat_after:
      x_concat = torch.cat((x, x_att), dim=-1)
      x = residual + self.concat_linear(x_concat)
    else:
      x = residual + self.dropout(x_att)
    if not self.normalize_before:
      x = self.norm_mha(x)

    # convolution module
    # Fake new cnn cache here, and then change it in conv_module
    new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
    if self.conv_module is not None:
      residual = x
      if self.normalize_before:
        x = self.norm_conv(x)
      x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
      x = residual + self.dropout(x)

      if not self.normalize_before:
        x = self.norm_conv(x)

    # feed forward module
    residual = x
    if self.normalize_before:
      x = self.norm_ff(x)

    x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
    if not self.normalize_before:
      x = self.norm_ff(x)

    if self.conv_module is not None:
      x = self.norm_final(x)

    return x, mask, new_att_cache, new_cnn_cache


class ConformerEncoder(torch.nn.Module):
  def __init__(
      self,
      input_size: int,
      output_size: int,
      attention_heads: int = 4,
      linear_units: int = 2048,
      num_blocks: int = 6,
      dropout_rate: float = 0.1,
      positional_dropout_rate: float = 0.1,
      attention_dropout_rate: float = 0.0,
      input_layer: str = "conv2d",
      pos_enc_layer_type: str = "rel_pos",
      normalize_before: bool = True,
      concat_after: bool = False,
      static_chunk_size: int = 0,
      use_dynamic_chunk: bool = False,
      global_cmvn: torch.nn.Module = None,
      use_dynamic_left_chunk: bool = False,
      positionwise_conv_kernel_size: int = 1,
      macaron_style: bool = True,
      selfattention_layer_type: str = "rel_selfattn",
      activation_type: str = "swish",
      use_cnn_module: bool = True,
      cnn_module_kernel: int = 15,
      causal: bool = False,
      cnn_module_norm: str = "batch_norm",
  ):
    """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            concat_after (bool): whether to concat attention layer's input
                and output.
                True: x -> x + linear(concat(x, att(x)))
                False: x -> x + att(x)
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
        """
    assert check_argument_types()
    logging.info("Init Conformer layers::")
    super().__init__()
    self._output_size = output_size

    if pos_enc_layer_type == "abs_pos":
      pos_enc_class = PositionalEncoding
    elif pos_enc_layer_type == "rel_pos":
      pos_enc_class = RelPositionalEncoding
    elif pos_enc_layer_type == "no_pos":
      pos_enc_class = NoPositionalEncoding
    else:
      raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)
    logging.info("Pos layers - {}".format(pos_enc_layer_type))

    if input_layer == "linear":
      subsampling_class = LinearNoSubsampling
    elif input_layer == "conv2d":
      subsampling_class = Conv2dSubsampling4
    elif input_layer == "conv2d8":
      subsampling_class = Conv2dSubsampling8
    else:
      raise ValueError("unknown input_layer: " + input_layer)
    logging.info("Input layers - {}".format(input_layer))

    self.global_cmvn = global_cmvn
    self.embed = subsampling_class(
      input_size,
      output_size,
      dropout_rate,
      pos_enc_class(output_size, positional_dropout_rate),
    )

    self.normalize_before = normalize_before
    self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
    self.static_chunk_size = static_chunk_size
    self.use_dynamic_chunk = use_dynamic_chunk
    self.use_dynamic_left_chunk = use_dynamic_left_chunk

    activation = get_activation(activation_type)

    # self-attention module definition
    if pos_enc_layer_type != "rel_pos":
      encoder_selfattn_layer = MultiHeadedAttention
    else:
      encoder_selfattn_layer = RelPositionMultiHeadedAttention
    encoder_selfattn_layer_args = (
      attention_heads,
      output_size,
      attention_dropout_rate,
    )
    logging.info("self attn args: {}".format(encoder_selfattn_layer_args))

    # feed-forward module definition
    positionwise_layer = PositionwiseFeedForward
    positionwise_layer_args = (
      output_size,
      linear_units,
      dropout_rate,
      activation,
    )
    logging.info("position-wise args: {}".format(positionwise_layer_args))

    # convolution module definition
    convolution_layer = ConvolutionModule
    convolution_layer_args = (output_size, cnn_module_kernel, activation,
                              cnn_module_norm, causal)

    self.encoders = torch.nn.ModuleList([
      ConformerEncoderLayer(
        output_size,
        encoder_selfattn_layer(*encoder_selfattn_layer_args),
        positionwise_layer(*positionwise_layer_args),
        positionwise_layer(
          *positionwise_layer_args) if macaron_style else None,
        convolution_layer(
          *convolution_layer_args) if use_cnn_module else None,
        dropout_rate,
        normalize_before,
        concat_after,
      ) for _ in range(num_blocks)
    ])
    return

  def output_size(self) -> int:
    return self._output_size

  def forward(
      self,
      xs: torch.Tensor,
      xs_lens: torch.Tensor,
      decoding_chunk_size: int = -1,
      num_decoding_left_chunks: int = -1,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        """
    T = xs.size(1)
    masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
    if self.global_cmvn is not None:
      xs = self.global_cmvn(xs)
    xs, pos_emb, masks = self.embed(xs, masks)
    mask_pad = masks  # (B, 1, T/subsample_rate)
    chunk_masks = add_optional_chunk_mask(xs, masks,
                                          self.use_dynamic_chunk,
                                          self.use_dynamic_left_chunk,
                                          decoding_chunk_size,
                                          self.static_chunk_size,
                                          num_decoding_left_chunks)
    for layer in self.encoders:
      xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
    if self.normalize_before:
      xs = self.after_norm(xs)
    # Here we assume the mask is not changed in encoder layers, so just
    # return the masks before encoder layers, and the masks will be used
    # for cross attention with decoder later
    return xs, masks

  def forward_chunk(
      self,
      xs: torch.Tensor,
      offset: int,
      required_cache_size: int,
      att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
      cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
      att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Forward just one chunk

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """
    assert xs.size(0) == 1
    # tmp_masks is just for interface compatibility
    tmp_masks = torch.ones(1,
                           xs.size(1),
                           device=xs.device,
                           dtype=torch.bool)
    tmp_masks = tmp_masks.unsqueeze(1)
    if self.global_cmvn is not None:
      xs = self.global_cmvn(xs)
    # NOTE(xcsong): Before embed, shape(xs) is (b=1, time, mel-dim)
    xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
    # NOTE(xcsong): After  embed, shape(xs) is (b=1, chunk_size, hidden-dim)
    elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
    chunk_size = xs.size(1)
    attention_key_size = cache_t1 + chunk_size
    pos_emb = self.embed.position_encoding(
      offset=offset - cache_t1, size=attention_key_size)
    if required_cache_size < 0:
      next_cache_start = 0
    elif required_cache_size == 0:
      next_cache_start = attention_key_size
    else:
      next_cache_start = max(attention_key_size - required_cache_size, 0)
    r_att_cache = []
    r_cnn_cache = []
    for i, layer in enumerate(self.encoders):
      # NOTE(xcsong): Before layer.forward
      #   shape(att_cache[i:i + 1]) is (1, head, cache_t1, d_k * 2),
      #   shape(cnn_cache[i])       is (b=1, hidden-dim, cache_t2)
      xs, _, new_att_cache, new_cnn_cache = layer(
        xs, att_mask, pos_emb,
        att_cache=att_cache[i:i + 1] if elayers > 0 else att_cache,
        cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache
      )
      # NOTE(xcsong): After layer.forward
      #   shape(new_att_cache) is (1, head, attention_key_size, d_k * 2),
      #   shape(new_cnn_cache) is (b=1, hidden-dim, cache_t2)
      r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
      r_cnn_cache.append(new_cnn_cache.unsqueeze(0))
    if self.normalize_before:
      xs = self.after_norm(xs)

    # NOTE(xcsong): shape(r_att_cache) is (elayers, head, ?, d_k * 2),
    #   ? may be larger than cache_t1, it depends on required_cache_size
    r_att_cache = torch.cat(r_att_cache, dim=0)
    # NOTE(xcsong): shape(r_cnn_cache) is (e, b=1, hidden-dim, cache_t2)
    r_cnn_cache = torch.cat(r_cnn_cache, dim=0)

    return (xs, r_att_cache, r_cnn_cache)

  def forward_chunk_by_chunk(
      self,
      xs: torch.Tensor,
      decoding_chunk_size: int,
      num_decoding_left_chunks: int = -1,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Forward input chunk by chunk with chunk_size like a streaming
            fashion

        Here we should pay special attention to computation cache in the
        streaming style forward chunk by chunk. Three things should be taken
        into account for computation in the current network:
            1. transformer/conformer encoder layers output cache
            2. convolution in conformer
            3. convolution in subsampling

        However, we don't implement subsampling cache for:
            1. We can control subsampling module to output the right result by
               overlapping input instead of cache left context, even though it
               wastes some computation, but subsampling only takes a very
               small fraction of computation in the whole model.
            2. Typically, there are several covolution layers with subsampling
               in subsampling module, it is tricky and complicated to do cache
               with different convolution layers with different subsampling
               rate.
            3. Currently, nn.Sequential is used to stack all the convolution
               layers in subsampling, we need to rewrite it to make it work
               with cache, which is not prefered.
        Args:
            xs (torch.Tensor): (1, max_len, dim)
            chunk_size (int): decoding chunk size
        """
    assert decoding_chunk_size > 0
    # The model is trained by static or dynamic chunk
    assert self.static_chunk_size > 0 or self.use_dynamic_chunk
    subsampling = self.embed.subsampling_rate
    context = self.embed.right_context + 1  # Add current frame
    stride = subsampling * decoding_chunk_size
    decoding_window = (decoding_chunk_size - 1) * subsampling + context
    num_frames = xs.size(1)
    att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
    cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0), device=xs.device)
    outputs = []
    offset = 0
    required_cache_size = decoding_chunk_size * num_decoding_left_chunks

    # Feed forward overlap input step by step
    for cur in range(0, num_frames - context + 1, stride):
      end = min(cur + decoding_window, num_frames)
      chunk_xs = xs[:, cur:end, :]
      (y, att_cache, cnn_cache) = self.forward_chunk(
        chunk_xs, offset, required_cache_size, att_cache, cnn_cache)
      outputs.append(y)
      offset += y.size(1)
    ys = torch.cat(outputs, 1)
    masks = torch.ones((1, 1, ys.size(1)), device=ys.device, dtype=torch.bool)
    return ys, masks


def _test_model():
  device = torch.device('cuda')
  model = ConformerEncoder(input_size=128, output_size=128,
                           linear_units=128).to(device)
  batch_size = 4
  feat_size = 128
  time_size = 1500
  x = torch.from_numpy(
    np.random.random([batch_size, time_size, feat_size])).float().to(
    device)
  xs_lens = torch.full(
    [x.size(0)], fill_value=x.size(1), dtype=torch.long).to(device).long()
  out, _ = model.forward(x, xs_lens=xs_lens, decoding_chunk_size=-1)
  print(out.size())
  return


if __name__ == '__main__':
  _test_model()
  pass
