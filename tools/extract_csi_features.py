#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2022/7/11 3:20 PM
# software: PyCharm

import argparse
import logging
import os
import random
import shutil
import subprocess

import librosa
import numpy as np

from src.dataset import SignalAug
from src.cqt import PyCqt
from src.utils import RARE_DELIMITER, load_hparams
from src.utils import read_lines, write_lines, line_to_dict, dict_to_line
from src.utils import remake_path_for_linux


def _sort_lines_by_utt(init_path, sorted_path):
  dump_lines = read_lines(init_path, log=False)
  dump_lines = sorted(dump_lines, key=lambda x: (line_to_dict(x)["utt"]))
  write_lines(sorted_path, dump_lines, log=True)
  return


def _remove_dup_line(init_path, new_path):
  logging.info("Remove line with same utt")
  old_line_num = len(read_lines(init_path, log=False))
  utt_set = set()
  valid_lines = []
  for line in read_lines(init_path, log=False):
    utt = line_to_dict(line)["utt"]
    if utt not in utt_set:
      utt_set.add(utt)
      valid_lines.append(line)
  logging.info("Filter stage: {}->{}".format(old_line_num, len(valid_lines)))
  write_lines(new_path, valid_lines)
  return


def _remove_invalid_line(init_path, new_path):
  old_line_num = len(read_lines(init_path, log=False))
  dump_lines = []
  for line in read_lines(init_path, log=False):
    local_data = line_to_dict(line)
    if not os.path.exists(local_data["wav"]):
      logging.info("Unvalid data for wav path: {}".format(line))
      continue
    dump_lines.append(line)
  logging.info("Filter stage: {}->{}".format(old_line_num, len(dump_lines)))
  write_lines(new_path, dump_lines)
  return


def _remove_line_with_same_dur(init_path, new_path):
  """remove line with same song-id and same dur-ms"""
  old_line_num = len(read_lines(init_path, log=False))
  dump_lines = []
  for line in read_lines(init_path, log=False):
    local_data = line_to_dict(line)
    if not os.path.exists(local_data["wav"]):
      logging.info("Unvalid data for wav path: {}".format(line))
      continue
    dump_lines.append(line)
  logging.info("Filter stage: {}->{}".format(old_line_num, len(dump_lines)))
  write_lines(new_path, dump_lines)
  return


def sox_change_speed(inp_path, out_path, k):
  cmd = "sox -q {} -t wav  -r 16000 -c 1 {} tempo {} " \
        "> sox.log 2> sox.log".format(
    remake_path_for_linux(inp_path), remake_path_for_linux(out_path), k)

  try:
    subprocess.call(cmd, shell=True)
    success = os.path.exists(out_path)
    if not success:
      logging.info("Error for sox: {}".format(cmd))
    return success
  except RuntimeError:
    logging.info("RuntimeError: {}".format(cmd))
    return False
  except EOFError:
    logging.info("EOFError: {}".format(cmd))
    return False


def _speed_aug(init_path, aug_speed_lst, aug_path, sp_dir):
  """add items with speed argument wav"""
  logging.info("speed factor: {}".format(aug_speed_lst))
  os.makedirs(sp_dir, exist_ok=True)
  total_lines = read_lines(init_path, log=False)
  dump_lines = []

  for line in total_lines:
    local_data = line_to_dict(line)
    wav_path = local_data["wav"]
    for speed in aug_speed_lst:
      if abs(speed - 1.0) > 0.01:
        sp_utt = "sp_{}-{}".format(speed, local_data["utt"])
        sp_wav_path = os.path.join(sp_dir, f"{sp_utt}.wav")
        if not os.path.exists(sp_wav_path):
          sox_change_speed(wav_path, sp_wav_path, speed)
      else:
        sp_utt = local_data["utt"]
        sp_wav_path = local_data["wav"]

      local_data["utt"] = sp_utt
      local_data["wav"] = sp_wav_path
      dump_lines.append(dict_to_line(local_data))
      if len(dump_lines) % 1000 == 0:
        logging.info("{}: {}".format(len(dump_lines), dump_lines[-1]))

  write_lines(aug_path, dump_lines)
  return


def _extract_cqt(init_path, out_path, cqt_dir):
  logging.info("Extract Cqt feature")
  os.makedirs(cqt_dir, exist_ok=True)

  py_cqt = PyCqt(sample_rate=16000, hop_size=0.04)

  dump_lines = []
  for line in read_lines(init_path, log=False):
    local_data = line_to_dict(line)
    wav_path = local_data["wav"]
    local_data["feat"] = os.path.join(cqt_dir,
                                      "{}.cqt.npy".format(local_data["utt"]))

    if not os.path.exists(local_data["feat"]):
      y, sr = librosa.load(wav_path, sr=16000)
      y = y / max(0.001, np.max(np.abs(y))) * 0.999
      cqt = py_cqt.compute_cqt(signal_float=y, feat_dim_first=False)
      np.save(local_data["feat"], cqt)
      local_data["feat_len"] = len(cqt)

    if "feat_len" not in local_data.keys():
      cqt = np.load(local_data["feat"])
      local_data["feat_len"] = len(cqt)

    dump_lines.append(dict_to_line(local_data))

    if len(dump_lines) % 1000 == 0:
      logging.info("Process cqt for {}items: {}".format(
        len(dump_lines), local_data["utt"]))

  write_lines(out_path, dump_lines)
  return


def _extract_cqt_with_noise(init_path, full_path, cqt_dir, hp_noise):
  logging.info("Extract Cqt feature with noise argumentation")
  os.makedirs(cqt_dir, exist_ok=True)

  py_cqt = PyCqt(sample_rate=16000, hop_size=0.04)
  sig_aug = SignalAug(hp_noise)
  vol_lst = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

  dump_lines = []
  for line in read_lines(init_path, log=False):
    local_data = line_to_dict(line)
    wav_path = local_data["wav"]
    local_data["utt"] = local_data["utt"] + "{}noise_{}".format(
      RARE_DELIMITER, hp_noise["name"])
    local_data["feat"] = os.path.join(cqt_dir,
                                      "{}.cqt.npy".format(local_data["utt"]))

    vol = random.choice(vol_lst)
    if not os.path.exists(local_data["feat"]):
      y, sr = librosa.load(wav_path, sr=16000)
      y = sig_aug.augmentation(y)
      y = y / max(0.001, np.max(np.abs(y))) * 0.999 * vol
      cqt = py_cqt.compute_cqt(signal_float=y, feat_dim_first=False)
      np.save(local_data["feat"], cqt)
      local_data["feat_len"] = len(cqt)

    if "feat_len" not in local_data.keys():
      cqt = np.load(local_data["feat"])
      local_data["feat_len"] = len(cqt)

    dump_lines.append(dict_to_line(local_data))

    if len(dump_lines) % 1000 == 0:
      logging.info("Process cqt for {}items: {}, vol:{}".format(
        len(dump_lines), local_data["utt"], vol))

  write_lines(full_path, dump_lines)
  return


def _add_song_id(init_path, out_path, map_path):
  """map format:: song_name->song_id"""
  song_id_map = {}
  dump_lines = []
  for line in read_lines(init_path, log=False):
    local_data = line_to_dict(line)
    song_name = local_data["song"]
    if song_name not in song_id_map.keys():
      song_id_map[song_name] = len(song_id_map)
    local_data["song_id"] = song_id_map[song_name]
    dump_lines.append(dict_to_line(local_data))
  write_lines(out_path, dump_lines)

  dump_lines = []
  for k, v in song_id_map.items():
    dump_lines.append("{} {}".format(k, v))
  write_lines(map_path, dump_lines)
  return


def _add_version_id(init_path, out_path):
  song_version_map = {}
  for line in read_lines(init_path, log=False):
    local_data = line_to_dict(line)
    song_id = local_data["song_id"]
    if song_id not in song_version_map.keys():
      song_version_map[song_id] = []
    song_version_map[song_id].append(local_data)

  dump_lines = []
  for k, v_lst in song_version_map.items():
    for version_id, local_data in enumerate(v_lst):
      local_data["version_id"] = version_id
      dump_lines.append(dict_to_line(local_data))
  write_lines(out_path, dump_lines)
  return


def _extract_song_num(full_path, song_name_map_path, song_id_map_path):
  """add map of song_id:num and song_name:num"""
  song_id_num = {}
  max_song_id = 0
  for line in read_lines(full_path):
    local_data = line_to_dict(line)
    song_id = local_data["song_id"]
    if song_id not in song_id_num.keys():
      song_id_num[song_id] = 0
    song_id_num[song_id] += 1
    if song_id >= max_song_id:
      max_song_id = song_id
  logging.info("max_song_id: {}".format(max_song_id))

  dump_data = list(song_id_num.items())
  dump_data = sorted(dump_data)
  dump_lines = ["{} {}".format(k, v) for k, v in dump_data]
  write_lines(song_id_map_path, dump_lines, log=False)

  song_num = {}
  for line in read_lines(full_path, log=False):
    local_data = line_to_dict(line)
    song_id = local_data["song"]
    if song_id not in song_num.keys():
      song_num[song_id] = 0
    song_num[song_id] += 1

  dump_data = list(song_num.items())
  dump_data = sorted(dump_data)
  dump_lines = ["{} {}".format(k, v) for k, v in dump_data]
  write_lines(song_name_map_path, dump_lines, log=False)
  return


def _sort_lines_by_song_id(full_path, sorted_path):
  dump_lines = read_lines(full_path, log=False)
  dump_lines = sorted(dump_lines,
                      key=lambda x: (int(line_to_dict(x)["song_id"]),
                                     int(line_to_dict(x)["version_id"])))
  write_lines(sorted_path, dump_lines, log=True)
  return


def _clean_lines(full_path, clean_path):
  dump_lines = []
  for line in read_lines(full_path):
    local_data = line_to_dict(line)
    clean_data = {
      "utt": local_data["utt"],
      "song_id": local_data["song_id"],
      "song": local_data["song"],
      "version_id": local_data["version_id"],
    }
    if "feat" in local_data.keys():
      clean_data.update({"feat_len": local_data["feat_len"],
                         "feat": local_data["feat"]})
    else:
      clean_data.update({"dur_ms": local_data["dur_ms"],
                         "wav": local_data["wav"]})
    dump_lines.append(dict_to_line(clean_data))
  write_lines(clean_path, dump_lines)
  return


def _generate_csi_features(hp, feat_dir, start_stage, end_stage):
  data_path = os.path.join(feat_dir, "dataset.txt")
  assert os.path.exists(data_path)

  init_path = os.path.join(feat_dir, "data.init.txt")
  shutil.copy(data_path, init_path)
  if start_stage <= 0 <= end_stage:
    logging.info("Stage 0: generate full path")
    _sort_lines_by_utt(init_path, init_path)
    _remove_dup_line(init_path, init_path)

  sp_aug_path = os.path.join(feat_dir, "sp_aug.txt")
  if start_stage <= 3 <= end_stage:
    logging.info("Stage 3: speed augmentation")
    if "aug_speed_mode" in hp.keys() and not os.path.exists(sp_aug_path):
      sp_dir = os.path.join(feat_dir, "sp_wav")
      _speed_aug(init_path, hp["aug_speed_mode"], sp_aug_path, sp_dir)

  full_path = os.path.join(feat_dir, "full.txt")
  if start_stage <= 4 <= end_stage:
    logging.info("Stage 4: extract cqt feature")
    cqt_dir = os.path.join(feat_dir, "cqt_feat")
    _extract_cqt(sp_aug_path, full_path, cqt_dir)

  hp_noise = hp.get("add_noise", None)
  if start_stage <= 5 <= end_stage and hp_noise and os.path.exists(hp_noise["noise_path"]):
    logging.info("Stage 5: add noise and extract cqt feature")
    noise_cqt_dir = os.path.join(feat_dir, "cqt_with_noise")
    _extract_cqt_with_noise(full_path, full_path, noise_cqt_dir,
                            hp_noise={"add_noise": hp_noise})

  if start_stage <= 8 <= end_stage:
    logging.info("Stage 8: add song_id")
    song_id_map_path = os.path.join(feat_dir, "song_id.map")
    if not os.path.exists(song_id_map_path):
      _add_song_id(full_path, full_path, song_id_map_path)

  if start_stage <= 9 <= end_stage:
    logging.info("Stage 9: add version_id")
    _add_version_id(full_path, full_path)

  if start_stage <= 10 <= end_stage:
    logging.info("Start stage 10: extract version num")
    song_id_map_path = os.path.join(feat_dir, "song_id_num.map")
    song_num_map_path = os.path.join(feat_dir, "song_name_num.map")
    _extract_song_num(full_path, song_num_map_path, song_id_map_path)

  if start_stage <= 11 <= end_stage:
    logging.info("Stage 11: clean for unused keys")
    _sort_lines_by_song_id(full_path, full_path)

  if start_stage <= 13 <= end_stage:
    logging.info("Stage 13:Split item to Train/Dev/Test")
  return


def _cmd():
  parser = argparse.ArgumentParser()
  parser.add_argument('feat_dir', help="feat_dir")
  parser.add_argument('--start_stage', type=int, default=0)
  parser.add_argument('--end_stage', type=int, default=100)
  args = parser.parse_args()
  hp_path = os.path.join(args.feat_dir, "hparams.yaml")
  hp = load_hparams(hp_path)
  print(hp)
  _generate_csi_features(hp, args.feat_dir, args.start_stage, args.end_stage)
  return


if __name__ == '__main__':
  _cmd()
  pass
