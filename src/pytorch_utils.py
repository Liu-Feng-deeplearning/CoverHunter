#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2021/3/26 10:20 AM
# software: PyCharm


import glob
import os

import torch


# todo: clean

def init_weights(m, mean=0.0, std=0.01):
  """ init weights with normal_ """
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
  return int((kernel_size * dilation - dilation) / 2)


def scan_last_checkpoint(cp_dir, prefix):
  pattern = os.path.join(cp_dir, prefix + '????????')
  cp_list = glob.glob(pattern)
  if len(cp_list) == 0:
    return None
  return sorted(cp_list)[-1]


def scan_and_load_checkpoint(cp_dir, prefix):
  pattern = os.path.join(cp_dir, prefix + '????????')
  cp_list = glob.glob(pattern)
  if len(cp_list) == 0:
    return None
  model_path = sorted(cp_list)[-1]
  checkpoint_dict = torch.load(model_path, map_location="cpu")
  print("Loading {}".format(model_path))
  return checkpoint_dict


def get_latest_model(hdf5_dir, prefix):
  model_path, last_epoch = None, 0
  for name in os.listdir(hdf5_dir):
    if name.startswith(prefix):
      epoch = int(name.replace("-", ".").replace("_", ".").split(".")[1])
      if epoch > last_epoch:
        last_epoch = epoch
        model_path = os.path.join(hdf5_dir, name)
  return model_path, last_epoch


def get_model_with_epoch(hdf5_dir, prefix, model_epoch):
  for name in os.listdir(hdf5_dir):
    if name.startswith(prefix):
      local_epoch = int(name.replace("-", ".").replace("_", ".").split(".")[1])
      if local_epoch == model_epoch:
        model_path = os.path.join(hdf5_dir, name)
        return model_path
  return None


def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']


def average_model(model_path_list, new_model_path):
  print(model_path_list)
  avg = None
  num = len(model_path_list)
  for path in model_path_list:
    print('Processing {}'.format(path))
    states = torch.load(path, map_location=torch.device('cpu'))
    if avg is None:
      avg = states
    else:
      for k in avg.keys():
        avg[k] += states[k]

  # average
  for k in avg.keys():
    if avg[k] is not None:
      avg[k] = torch.true_divide(avg[k], num)

  print('Saving to {}'.format(new_model_path))
  torch.save(avg, new_model_path)
  return


if __name__ == '__main__':
  pass
