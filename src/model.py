#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import os
from typing import Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.module.conformer import ConformerEncoder
from src.module.layers import Conv1d, Linear
from src.basic_model import BasicModel
from src.loss import HardTripletLoss, FocalLoss, CenterLoss
from src.utils import load_hparams


class AttentiveStatisticsPooling(torch.nn.Module):
  """This class implements an attentive statistic pooling layer for each channel.
  It returns the concatenated mean and std of the input tensor.

  Arguments
  ---------
  channels: int
      The number of input channels.
  output_channels: int
      The number of output channels.
  """

  def __init__(self, channels, output_channels):
    super().__init__()

    self._eps = 1e-12
    self._linear = Linear(channels * 3, channels)
    self._tanh = torch.nn.Tanh()
    self._conv = Conv1d(
      in_channels=channels, out_channels=channels, kernel_size=1
    )
    self._final_layer = torch.nn.Linear(channels * 2, output_channels,
                                        bias=False)
    logging.info("Init AttentiveStatisticsPooling with {}->{}".format(
      channels, output_channels))
    return

  @staticmethod
  def _compute_statistics(x: torch.Tensor,
                          m: torch.Tensor,
                          eps: float,
                          dim: int = 2):
    mean = (m * x).sum(dim)
    std = torch.sqrt((m * (x - mean.unsqueeze(dim)).pow(2)).sum(dim).clamp(eps))
    return mean, std

  def forward(self, x: torch.Tensor):
    """Calculates mean and std for a batch (input tensor).

    Args:
      x : torch.Tensor
          Tensor of shape [N, L, C].
    """

    x = x.transpose(1, 2)
    L = x.shape[-1]
    lengths = torch.ones(x.shape[0], device=x.device)
    mask = self.length_to_mask(lengths * L, max_len=L, device=x.device)
    mask = mask.unsqueeze(1)
    total = mask.sum(dim=2, keepdim=True).float()

    mean, std = self._compute_statistics(x, mask / total, self._eps)
    mean = mean.unsqueeze(2).repeat(1, 1, L)
    std = std.unsqueeze(2).repeat(1, 1, L)
    attn = torch.cat([x, mean, std], dim=1)
    attn = self._conv(self._tanh(self._linear(
      attn.transpose(1, 2)).transpose(1, 2)))

    attn = attn.masked_fill(mask == 0, float("-inf"))  # Filter out zero-padding
    attn = F.softmax(attn, dim=2)
    mean, std = self._compute_statistics(x, attn, self._eps)
    pooled_stats = self._final_layer(torch.cat((mean, std), dim=1))
    return pooled_stats

  def forward_with_mask(self, x: torch.Tensor,
                        lengths: Optional[torch.Tensor] = None):
    """Calculates mean and std for a batch (input tensor).

    Args:
      x : torch.Tensor
          Tensor of shape [N, C, L].
      lengths:
    """
    L = x.shape[-1]

    if lengths is None:
      lengths = torch.ones(x.shape[0], device=x.device)

    # Make binary mask of shape [N, 1, L]
    mask = self.length_to_mask(lengths * L, max_len=L, device=x.device)
    mask = mask.unsqueeze(1)

    # Expand the temporal context of the pooling layer by allowing the
    # self-attention to look at global properties of the utterance.

    # torch.std is unstable for backward computation
    # https://github.com/pytorch/pytorch/issues/4320
    total = mask.sum(dim=2, keepdim=True).float()
    mean, std = self._compute_statistics(x, mask / total, self._eps)
    print("zz:", mean.size())

    mean = mean.unsqueeze(2).repeat(1, 1, L)
    std = std.unsqueeze(2).repeat(1, 1, L)
    print("zz:", mean.size())
    attn = torch.cat([x, mean, std], dim=1)

    # Apply layers
    attn = self.conv(self._tanh(self._linear(attn, lengths)))

    # Filter out zero-paddings
    attn = attn.masked_fill(mask == 0, float("-inf"))

    attn = F.softmax(attn, dim=2)
    mean, std = self._compute_statistics(x, attn, self._eps)
    # Append mean and std of the batch
    pooled_stats = torch.cat((mean, std), dim=1)
    pooled_stats = pooled_stats.unsqueeze(2)
    return pooled_stats

  @staticmethod
  def length_to_mask(length: torch.Tensor,
                     max_len: Optional[int] = None,
                     dtype: Optional[torch.dtype] = None,
                     device: Optional[torch.device] = None):
    """Creates a binary mask for each sequence.

    Arguments
    ---------
    length : torch.LongTensor
        Containing the length of each sequence in the batch. Must be 1D.
    max_len : int
        Max length for the mask, also the size of the second dimension.
    dtype : torch.dtype, default: None
        The dtype of the generated mask.
    device: torch.device, default: None
        The device to put the mask variable.

    Returns
    -------
    mask : tensor
        The binary mask.

    Example
    -------
    """
    assert len(length.shape) == 1

    if max_len is None:
      max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
      max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
      dtype = length.dtype

    if device is None:
      device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask


class Model(BasicModel):
  """ Csi Backbone Model:
      Batch-norm
      Conformer-Encoder(x6 or x4)
      Global-Avg-Pool
      Resize and linear
      Bn-neck

  Loss:
    Focal-loss(ce) + Triplet-loss + Center-loss

  Args:
    hp: Dict with all parameters
  """

  def __init__(self, hp: Dict):
    super(BasicModel, self).__init__()
    self._hp = hp

    self._global_cmvn = torch.nn.BatchNorm1d(hp["input_dim"])
    self._encoder = ConformerEncoder(
      input_size=hp["input_dim"],
      output_size=hp["encoder"]["output_dims"],
      linear_units=hp["encoder"]["attention_dim"],
      num_blocks=hp["encoder"]["num_blocks"])

    if hp["encoder"]["output_dims"] != hp["embed_dim"]:
      self._embed_lo = torch.nn.Linear(hp["encoder"]["output_dims"],
                                       hp["embed_dim"])
    else:
      self._embed_lo = None

    # Bottleneck
    self._bottleneck = torch.nn.BatchNorm1d(hp["embed_dim"])
    self._bottleneck.bias.requires_grad_(False)  # no shift

    self._pool_layer = AttentiveStatisticsPooling(
      hp["embed_dim"], output_channels=hp["embed_dim"])
    self._ce_layer = torch.nn.Linear(
      hp["embed_dim"], hp["ce"]["output_dims"], bias=False)

    # Loss
    if "alpha" in hp["ce"].keys():
      alpha = np.load(hp["ce"]["alpha"])
      alpha = 1.0 / (alpha + 1)
      alpha = alpha / np.sum(alpha)
      logging.warning("use alpha with {}".format(len(alpha)))
    else:
      alpha = None
      logging.warning("Not use alpha")

    self._ce_loss = FocalLoss(alpha=alpha, gamma=self._hp["ce"]["gamma"],
                              num_cls=self._hp["ce"]["output_dims"])
    self._triplet_loss = HardTripletLoss(margin=hp["triplet"]["margin"])
    self._center_loss = CenterLoss(num_classes=self._hp["ce"]["output_dims"],
                                   feat_dim=hp["embed_dim"], use_gpu=True)
    logging.info(
      "Model size: {:.3f}M \n".format(self.model_size() / 1000 / 1000))
    return

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """feat[b, frame_size, feat_size] -> embed[b, embed_dim]"""

    x = self._global_cmvn(x.transpose(1, 2)).transpose(1, 2)
    xs_lens = torch.full(
      [x.size(0)], fill_value=x.size(1), dtype=torch.long).to(x.device).long()
    x, _ = self._encoder(x, xs_lens=xs_lens, decoding_chunk_size=-1)
    x = self._pool_layer(x)
    return x

  @torch.jit.ignore
  def compute_loss(self, anchor: torch.Tensor, label: torch.Tensor) -> Tuple[
    torch.Tensor, Dict]:
    """compute ce and triplet loss"""
    f_t = self.forward(anchor)
    # print("f_t", f_t)

    f_i = self._bottleneck(f_t)
    ce_pred = self._ce_layer(f_i)
    # print("ce_pred", ce_pred)
    ce_loss = self._ce_loss(ce_pred, label)
    loss = ce_loss * self._hp["ce"]["weight"]
    loss_dict = {"ce_loss": ce_loss}

    tri_weight = self._hp["triplet"]["weight"]
    if tri_weight > 0.01:
      tri_loss = self._triplet_loss(f_t, label)
      loss = loss + tri_loss * tri_weight
      loss_dict.update({"tri_loss": tri_loss})

    cen_weight = self._hp["center"]["weight"]
    if cen_weight > 0.0:
      cen_loss = self._center_loss(f_t, label)
      loss = loss + cen_loss * cen_weight
      loss_dict.update({"cen_loss": cen_loss})
    return loss, loss_dict

  @torch.jit.ignore
  def inference(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
      embed = self.forward(feat)
      embed_ce = self._ce_layer(embed)
    return embed, embed_ce

  @torch.jit.export
  def compute_embed(self, feat: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      embed = self.forward(feat)
    return embed


def _test_model():
  device = torch.device('cuda')
  model_dir = "/workspace/project-nas-10487/liufeng/finger_print/egs/" \
              "pop_ch_fast/v16f"
  hp_path = os.path.join(model_dir, "config/hparams.yaml")
  hp = load_hparams(hp_path)
  model = Model(hp).float().to(device)
  batch_size = 4
  feat_size = hp["input_dim"]
  time_size = 1500
  feat = torch.from_numpy(
    np.random.random([batch_size, time_size, feat_size])).float().to(
    device)
  label = torch.from_numpy(2000 * np.random.random([batch_size])).long().to(
    device)
  loss = model.compute_loss(feat, label)
  print("loss:", loss)

  model = Model(hp, dump=True).to("cpu")
  model.dump_torch_script(
    "/workspace/project-nas-10487/liufeng/finger_print/"
    "aigit-runtime-finger-print/build_csi/xx.pt")
  return


if __name__ == '__main__':
  _test_model()
  pass
