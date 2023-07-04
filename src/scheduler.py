#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2021/7/7 10:24 AM
# software: PyCharm


import warnings

import torch


class UserDefineExponentialLR(torch.optim.lr_scheduler._LRScheduler):
  """Decays the learning rate of each parameter group by _gamma every epoch.
  When last_epoch=-1, sets initial lr as lr.

  Args:
      optimizer (Optimizer): Wrapped optimizer.
      gamma (float): Multiplicative factor of learning rate decay.
      min_lr(float): min lr.
      last_epoch (int): The index of last epoch. Default: -1.

  """

  def __init__(self, optimizer, gamma, min_lr, last_epoch=-1, warmup=False,
               warmup_steps=5000):
    self.gamma = gamma
    self.min_lr = min_lr
    self._warmup = warmup
    super(UserDefineExponentialLR, self).__init__(optimizer, last_epoch)

    if warmup:
      print("Using Warmup for Learning: {}".format(warmup_steps))
      self._warmup_steps = warmup_steps
      self.get_lr()

  def get_lr(self):
    if not self._get_lr_called_within_step:
      warnings.warn(
        "To get the last learning rate computed by the scheduler, "
        "please use `get_last_lr()`.", UserWarning)

    if self.last_epoch == 0:
      return self.base_lrs

    if not self._warmup:
      lr = [group['lr'] * self.gamma for group in self.optimizer.param_groups]
      lr[0] = lr[0] if lr[0] > self.min_lr else self.min_lr
    else:
      local_step = self.optimizer._step_count
      lr = [group['lr'] for group in self.optimizer.param_groups]
      if local_step <= self._warmup_steps + 1:
        lr[0] = self.base_lrs[0] * local_step / self._warmup_steps
        # print("debug:", self.base_lrs[0], local_step / self._warmup_steps, lr[0])
      else:
        lr[0] = lr[0] * self.gamma
        lr[0] = lr[0] if lr[0] > self.min_lr else self.min_lr
    return lr

  def _get_closed_form_lr(self):
    return [base_lr * self.gamma ** self.last_epoch
            for base_lr in self.base_lrs]
