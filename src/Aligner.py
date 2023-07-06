#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: liufeng
# datetime: 2023/7/5 5:11 PM

"""coarse-to-fine alignment as mentioned at https://arxiv.org/abs/2306.09025"""

import os

import numpy as np

from src.utils import read_lines, write_lines, line_to_dict, dict_to_line


class Aligner:
  def __init__(self, embed_dir):
    self._embed_dir = embed_dir
    self._memory_data = dict()
    for name in os.listdir(embed_dir):
      data_path = os.path.join(embed_dir, name)
      embed = np.load(data_path)
      utt_name = name.replace(".npy", "")
      self._memory_data[utt_name] = embed
    return

  def _get_shift_frame(self, data_i, data_j):
    assert data_i["song"] == data_j["song"]
    utt_i = data_i["utt"]
    utt_j = data_j["utt"]
    if utt_i == utt_j:
      return 0

    idx_npy_i = dict()
    for k, v in self._memory_data.items():
      if utt_i in k:
        idx_i = int(k.replace("start-", ".").split(".")[1])
        idx_npy_i[idx_i] = v

    idx_npy_j = dict()
    for k, v in self._memory_data.items():
      if k.startswith(utt_j):
        idx_j = int(k.replace("start-", ".").split(".")[1])
        idx_npy_j[idx_j] = v

    res = []
    for idx_i, npy_i in idx_npy_i.items():
      all_distance = []
      for idx_j, npy_j in idx_npy_j.items():
        if idx_i != idx_j and utt_i != utt_j:
          all_distance.append((idx_j, np.linalg.norm(npy_i - npy_j)))
      all_distance = sorted(all_distance, key=lambda x: x[1])
      nearest_j = all_distance[0][0]
      res.append((idx_i, nearest_j, idx_i - nearest_j))

    count_delta = dict()
    for _, _, z in res:
      if z not in count_delta.keys():
        count_delta[z] = 0
      count_delta[z] += 1
    count_delta_lst = list(count_delta.items())
    count_delta_lst = sorted(count_delta_lst, key=lambda x: x[1], reverse=True)
    shift_frame, _ = count_delta_lst[0]
    return shift_frame

  def align(self, data_path, output_path):
    dump_lines = []
    data_lines = read_lines(data_path)
    for line_i in data_lines:
      local_data_i = line_to_dict(line_i)
      for line_j in data_lines:
        local_data_j = line_to_dict(line_j)
        if local_data_i["song_id"] == local_data_j["song_id"]:
          shift_frame = self._get_shift_frame(local_data_i, local_data_j)
          dump_lines.append(dict_to_line(
            {"utt_i": local_data_i["utt"], "utt_j": local_data_j["utt"],
             "shift_frame_i_to_j": shift_frame}))
    write_lines(output_path, dump_lines)
    return
