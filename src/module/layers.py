#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2022/7/18 8:00 PM
# software: PyCharm

import torch


class Linear(torch.nn.Module):
  def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
    super(Linear, self).__init__()
    self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

    torch.nn.init.xavier_uniform_(
      self.linear_layer.weight,
      gain=torch.nn.init.calculate_gain(w_init_gain))
    return

  def forward(self, x):
    return self.linear_layer(x)


class Conv1d(torch.nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
               padding=None, dilation=1, bias=True, w_init_gain='linear'):
    super(Conv1d, self).__init__()
    if padding is None:
      assert (kernel_size % 2 == 1)
      padding = int(dilation * (kernel_size - 1) / 2)

    self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation,
                                bias=bias)

    torch.nn.init.xavier_uniform_(
      self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
    return

  def forward(self, signal):
    conv_signal = self.conv(signal)
    return conv_signal
