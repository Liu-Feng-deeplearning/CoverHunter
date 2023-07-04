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


# Code below is for Test and Debug
def train(_optim_g):
  _optim_g.step()
  return


def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']


def test_for_schedule():
  from src.utils.utils import load_hparams
  from src.csi.v17.model import Model
  hp_path = "/workspace/project-nas-10487/liufeng/finger_print/egs/" \
            "pop_ch_fast/v17a_1024/config/hparams.yaml"
  hp = load_hparams(hp_path)

  model = Model(hp=hp).float().to("cpu")
  print("model size: {:.3f}M".format(model.model_size() / 1000 / 1000))

  _optim_g = torch.optim.AdamW(model.parameters(), hp["learning_rate"],
                               betas=[hp["adam_b1"], hp["adam_b2"]])
  _scheduler_g = UserDefineExponentialLR(
    _optim_g, gamma=0.99, min_lr=0.00001, last_epoch=-1, warmup=hp["warmup"],
    warmup_steps=hp["warmup_steps"])
  print("XX:", get_lr(_optim_g), -1)
  _scheduler_g.step()
  print("XX:", get_lr(_optim_g), -1)

  exit()
  for step in range(360 * 1000):
    # _optim_g.step()
    train(_optim_g)
    if step % 1000 == 0:
      _scheduler_g.step()
      # if step % 5000 == 0:
      print("step:{}, lr:{}, {}".format(step, _scheduler_g.get_lr(),
                                        get_lr(_optim_g)))
  return


if __name__ == '__main__':
  test_for_schedule()
