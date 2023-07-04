#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import math
import random
import subprocess

import librosa
import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from torch.utils.data.sampler import Sampler

from src.feature.cqt import shorter
from src.utils import read_lines, line_to_dict

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

random.seed(1)


def _float_signal_to_int16(signal):
  signal = signal * 32768
  signal = np.int16(signal)
  return signal


class SignalAug:
  """Signal Augmentation, not change signal shape except "_change_speed" function

  Notes:
    input should be numpy.array of int16[-32768, 32767] or float[0, 1]
    output will be numpy.array float[0, 1]
  """

  def __init__(self, hp):
    self._hp = hp
    if "add_noise" in self._hp.keys():
      self._noise_path_lst = read_lines(hp["add_noise"]["noise_path"],
                                        log=False)
      logging.info("Noise Data path:{}".format(hp["add_noise"]["noise_path"]))
      logging.info("Noise Data items:{}".format(len(self._noise_path_lst)))
    else:
      self._noise_path_lst = None
    if "seed" not in hp.keys():
      hp["seed"] = 1234
    random.seed(hp["seed"])
    np.random.seed(hp["seed"])
    logging.info("SignalAug hparams: {}".format(hp))
    return

  @staticmethod
  def _change_volume(signal, coef):
    """ change signal magnitude with coef

    Args:
      signal:
      coef: float from 0 to 1

    Returns:

    """
    max_val = np.max(np.abs(signal)) + 0.01
    signal = signal / max_val * 0.999
    return signal * coef

  @staticmethod
  def _add_noise(signal, noise, snr):
    """ add noise to signal with snr

    Args:
      signal: numpy
      noise: numpy
      snr: float, log-signal-to-noise-ratio, smaller means more noise

    Returns:
      noise

    """

    signal = signal / max(0.001, np.max(np.abs(signal))) * 0.495
    noise = noise / max(0.001, np.max(np.abs(noise))) * 0.495
    snr = 10 ** (snr / 10.0)
    coef = np.sqrt(1 / snr)
    new_signal = signal + noise * coef
    new_signal = new_signal / max(0.001, np.max(np.abs(new_signal))) * 0.95
    return new_signal

  @staticmethod
  def _change_tempo(signal, coef):
    """

    Args:
      signal:
      coef:

    Returns:

    Notes:
      length of signal will be changed
    """
    args = ["sox", "-t", "s16", "-r", "16000", "-b", "16", "-c", "1", "-",
            "-r", "16000", "-t", "raw", "-",
            "tempo", "-s", coef]
    args = [str(x) for x in args]
    logging.info("cmd:{}".format(" ".join(args)))

    process_handle = subprocess.Popen(
      args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
    x = np.int16(signal)
    out, err = process_handle.communicate(x.T.tobytes(order='F'))
    out = np.frombuffer(out, dtype=np.int16)

    status = process_handle.returncode
    if status > 0:
      logging.info("status:{}, err:{}".format(status, err.decode("utf-8")))

    return out

  @staticmethod
  def _change_speed(signal, coef):
    """

    Args:
      signal:
      coef:

    Returns:

    Notes:
      length of signal will be changed
    """
    args = ["sox", "-t", "s16", "-r", "16000", "-b", "16", "-c", "1", "-",
            "-r", "16000", "-t", "raw", "-",
            "speed", coef]
    args = [str(x) for x in args]
    logging.info("cmd:{}".format(" ".join(args)))

    process_handle = subprocess.Popen(
      args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
    x = _float_signal_to_int16(signal)
    out, err = process_handle.communicate(x.T.tobytes(order='F'))
    out = np.frombuffer(out, dtype=np.int16)

    status = process_handle.returncode
    if status > 0:
      logging.info("status:{}, err:{}".format(status, err.decode("utf-8")))
    signal = out
    return signal

  @staticmethod
  def _change_pitch(signal, pitch_factor):
    """ wrapper for sox

    Args:
      signal: numpy.array
      pitch_factor: float, means ratio of new pitch / old pitch

    Returns:
      augmented signal

    Notes:
      Very slow

    References:
      https://github.com/rabitt/pysox/blob/master/sox/transform.py
    """
    semitone = math.log(pitch_factor, 2) * 12
    cents_of_semitone = semitone * 100
    args = ["sox", "-t", "s16", "-r", "16000", "-b", "16", "-c", "1", "-",
            "-r", "16000", "-t", "raw", "-",
            "pitch", cents_of_semitone]
    args = [str(x) for x in args]

    process_handle = subprocess.Popen(
      args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
    )
    x = _float_signal_to_int16(signal)
    out, err = process_handle.communicate(x.T.tobytes(order='F'))
    if process_handle.returncode != 0:
      return signal

    out = np.frombuffer(out, dtype=np.int16)
    # print("status:", status)
    # print("err:", err.decode("utf-8"))
    return out

  def augmentation(self, signal):
    """vanilla signal[N] -> augmentation signal[N]"""
    signal = signal.astype(float)
    if "speed" in self._hp.keys():
      if random.random() <= self._hp["speed"]["prob"]:
        coef = random.choice(self._hp["speed"]["coef"])
        signal = self._change_speed(signal, coef)

    if "tempo" in self._hp.keys():
      if random.random() <= self._hp["tempo"]["prob"]:
        coef = random.choice(self._hp["tempo"]["coef"])
        signal = self._change_tempo(signal, coef)

    if "pitch" in self._hp.keys():
      if random.random() <= self._hp["pitch"]["prob"]:
        coef = random.choice(self._hp["pitch"]["shift"])
        signal = self._change_pitch(signal, coef)

    if "add_noise" in self._hp.keys() and self._hp["add_noise"]["prob"] > 0.01:
      hp_noise = self._hp["add_noise"]
      noise_data = line_to_dict(random.choice(self._noise_path_lst))
      noise_sig_init, _ = librosa.load(noise_data["wav"], sr=hp_noise["sr"])
      noise_signal = noise_sig_init[random.randint(0, len(noise_sig_init)):]
      while len(noise_signal) < len(signal):
        noise_signal = np.concatenate([noise_signal, noise_sig_init])

      to_add_noise = np.zeros_like(signal)
      chunk = hp_noise["sr"] * hp_noise["chunk"]
      for i in range(int(len(to_add_noise) / chunk)):
        if random.random() <= hp_noise["prob"]:
          to_add_noise[i * chunk: (i + 1) * chunk] = noise_signal[
                                                     i * chunk: (i + 1) * chunk]

      snr = random.choice([-30, -20, -10, 0, 10])
      signal = self._add_noise(signal, noise=to_add_noise, snr=snr)

    if "volume" in self._hp.keys():
      if random.random() <= self._hp["volume"]["prob"]:
        coef = random.random()
        signal = self._change_volume(signal, coef)
    return signal


class SpecAug:
  """Spec Augmentation(Do not change data shape)"""

  def __init__(self, hp=None, logger=None):
    self._hp = hp
    if "seed" not in hp.keys():
      hp["seed"] = 1234
    random.seed(hp["seed"])
    np.random.seed(hp["seed"])
    if logger:
      logger.info("SpecAug hparams {}".format(hp))
    return

  @staticmethod
  def _mask_silence(feat, p_threshold=0.1):
    if random.random() > p_threshold:
      return feat

    w, h = np.shape(feat)
    feat_aug = feat
    for i in range(w):
      if random.random() < 0.1:
        feat_aug[i, :] = 0
    for i in range(h):
      if random.random() < 0.1:
        feat_aug[:, i] = 0
    return feat

  @staticmethod
  def _duplicate(feat, p_threshold=0.1):
    if random.random() > p_threshold:
      return feat

    w, h = np.shape(feat)
    feat_aug = feat
    for i in range(1, w):
      if random.random() < 0.1:
        feat_aug[i, :] = feat_aug[i - 1, :]
    for i in range(1, h):
      if random.random() < 0.1:
        feat_aug[:, i] = feat_aug[:, i - 1]
    return feat

  @staticmethod
  def _roll(feat, shift_num=12):
    w, h = np.shape(feat)
    shift_amount = random.randint(-1, 1) * shift_num
    for i in range(w):
      feat[i, :] = np.roll(feat[i, :], shift_amount)
    return feat

  @staticmethod
  def _random_erase(feat, region_num=4, region_size=(0.25, 0.1),
                    region_val=-80):
    w, h = np.shape(feat)
    region_w = int(w * region_size[0])
    region_h = int(h * region_size[1])
    for _ in range(region_num):
      center_w = int(random.random() * (w - region_w))
      center_h = int(random.random() * (h - region_h))
      feat[center_w - region_w // 2:center_w + region_w // 2,
      center_h - region_h // 2:center_h + region_h // 2] = region_val
    return feat

  def augmentation(self, feat):
    """vanilla feature[80 x N] -> augmentation feature[80 x N]"""
    if "roll_pitch" in self._hp.keys():
      p = random.random()
      if p <= self._hp["roll_pitch"]["prob"]:
        feat = self._roll(feat, shift_num=self._hp["roll_pitch"]["shift_num"])

    if "random_erase" in self._hp.keys():
      p = random.random()
      if p <= self._hp["random_erase"]["prob"]:
        feat = self._random_erase(feat, region_val=random.random() * (-80),
                                  region_num=self._hp["random_erase"][
                                    "erase_num"])
    return feat


class AudioFeatDataset(torch.utils.data.Dataset):
  """Simple DataSet, to get chunk data one by one. if feat less than length, it
  will be padding with zero.

  mode:
    - random: cut chunk from feat from random start
    - defined: cut feat with "start/chunk_len" info from line
  """

  def __init__(self, hp, data_path=None, data_lines=None, train=False,
               mode="random", chunk_len=None, logger=None):

    if data_path:
      data_lines = read_lines(data_path, log=False)

    self._hp = hp
    self._data = []
    self._mode = mode

    assert mode in ["random", "defined"], "invalid mode: {}".format(mode)
    if mode == "random":
      for line in data_lines:
        local_data = line_to_dict(line)
        self._data.append((local_data["utt"], local_data["song_id"],
                           local_data["feat"], local_data["feat_len"],
                           chunk_len))
    elif mode == "defined":
      for line in data_lines:
        local_data = line_to_dict(line)
        if "start" not in local_data.keys():
          local_data["start"] = 0
        self._data.append((local_data["utt"], local_data["song_id"],
                           local_data["feat"], local_data["start"],
                           chunk_len))
    else:
      raise Exception("invalid mode".format())

    if logger:
      logger.info(
        'Init AudioFeatDataset with mode-{}, chunk_len-{}'.format(
          mode, chunk_len))
      logger.info('Input dataset items: {}, valid items: {}'.format(
        len(data_lines), self.__len__()))

    if train and "spec_augmentation" in hp.keys():
      self._aug = SpecAug(hp["spec_augmentation"], logger)
    else:
      self._aug = None
      if logger:
        logger.info("No spec_augmentation!")

    return

  def __len__(self):
    return len(self._data)

  def __getitem__(self, idx):
    utt, label, feat_path, len_or_start, chunk_len = self._data[idx]

    if self._mode == "random":
      feat_len = len_or_start
      if (feat_len - chunk_len) > 0:
        start = int(random.random() * (feat_len - chunk_len))
      else:
        start = 0
    elif self._mode == "defined":
      utt, label, feat_path, start, chunk_len = self._data[idx]
    else:
      raise Exception("invalid mode".format())

    feat = np.load(feat_path)
    feat = feat[start:start + chunk_len]
    if len(feat) < chunk_len:
      feat = np.pad(feat, pad_width=((0, chunk_len - len(feat)), (0, 0)),
                    mode="constant", constant_values=-100)

    feat = shorter(feat, self._hp["mean_size"])
    if self._aug:
      feat = self._aug.augmentation(feat)

    feat = torch.from_numpy(feat)
    label = torch.tensor(label).long()
    return utt, feat, label


class MPerClassSampler(Sampler):
  """ At every iteration, this will return m samples per class. For example,
      if dataloader's batch-size is 100, and m = 5, then 20 classes with 5
      samples iter will be returned

      Support distribute with set distribute as True. All samples will be
      put all ranks randomly, but samples with same label will be at same gpu
      for contrastive loss.

      Notes:
        After every epoch, set_epoch should be called, and data index of every
        ranks will be changed. Random of index will accept epoch number as seed,
        to make sure every ranks has not override data.
  """

  def __init__(self, data_path, m, batch_size, distribute=False, logger=None):
    data_lines = read_lines(data_path, log=False)
    if distribute:
      if not dist.is_available():
        raise RuntimeError("Requires distributed package to be available")
      total_rank = dist.get_world_size()
      local_rank = dist.get_rank()
    else:
      total_rank = -1
      local_rank = -1

    self._m_per_class = m
    self._batch_size = batch_size
    self._logger = logger

    self._labels_to_indices = self._get_labels_to_indices(data_lines)
    self._global_labels = list(self._labels_to_indices.keys())

    if not distribute:
      self.labels = self._global_labels
    else:
      self.labels = self._split_label_randoms(seed=0)[local_rank]

    assert (self._batch_size % self._m_per_class) == 0, \
      "m_per_class must divide batch_size without any remainder"

    self._sample_length = self._get_sample_length()

    if logger:
      if not distribute:
        logger.info(
          "Init Sampler with Mper with {} items, and m = {}, batch_num = {}"
          "\n".format(self._sample_length, m, self.num_iters()))
      else:
        logger.info(
          "Init Sampler with Mper with {} items, and m = {}, "
          "rank = {}, num_replicas = {}, batch_num = {}"
          "\n".format(self._sample_length, m, local_rank, total_rank,
                      self.num_iters()))
    return

  def __iter__(self):
    idx_list = [0] * self._sample_length
    i = 0
    num_iters = self.num_iters()
    for _ in range(num_iters):
      random.shuffle(self.labels)
      curr_label_set = self.labels[: self._batch_size // self._m_per_class]
      for label in curr_label_set:
        t = self._labels_to_indices[label].copy()
        random.shuffle(t)
        idx_list[i: i + self._m_per_class] = t[:self._m_per_class]
        i += self._m_per_class
    return iter(idx_list)

  def shuffle_data_on_ranks(self, seed, display_details=True):
    local_rank = dist.get_rank()
    total_rank = dist.get_world_size()
    self.labels = self._split_label_randoms(seed)[local_rank]
    self._sample_length = self._get_sample_length()
    if self._logger and display_details:
      display = sorted(self.labels)[:5]
      self._logger.info(
        "Shuffle sampler with {} items, rank = {}, total rank = {}, "
        "batch_num = {}, label of head{} is {}".format(
          self._sample_length, local_rank, total_rank, self.num_iters(),
          len(display), display))
    return

  def num_iters(self):
    return self._sample_length // self._batch_size

  def _get_sample_length(self):
    sample_length = sum([len(self._labels_to_indices[k]) for k in self.labels])
    sample_length -= sample_length % self._batch_size
    return sample_length

  def _split_label_randoms(self, seed):
    split_label = []
    global_label = [x for x in self._labels_to_indices.keys()].copy()
    random.Random(seed).shuffle(global_label)
    for i in range(dist.get_world_size()):
      split_label.append(global_label[i::dist.get_world_size()])
    return split_label

  # @staticmethod
  def _get_labels_to_indices(self, data_lines, distribute=False):
    """ Creates _labels_to_indices, which is a dictionary mapping each label
    to list of indices that will be used to index into dataset.

    Notes: sum of indices must be equal to len(dataset)
    """
    labels_to_indices = {}
    for index, line in enumerate(data_lines):
      local_data = line_to_dict(line)
      label = local_data["song_id"]
      if distribute:
        total_rank = dist.get_world_size()
        local_rank = dist.get_rank()
        if label % total_rank != local_rank:
          continue

      if label not in labels_to_indices.keys():
        labels_to_indices[label] = []

      labels_to_indices[label].append(index)

    for k in labels_to_indices.keys():
      expand_indices = labels_to_indices[k].copy()
      while len(expand_indices) < self._m_per_class:
        expand_indices.extend(labels_to_indices[k])
      labels_to_indices[k] = expand_indices.copy()
    return labels_to_indices

  def __len__(self):
    return self._sample_length
