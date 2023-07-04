#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2021/11/23 4:44 PM
# software: PyCharm


import json
import logging
import os
import shutil

import numpy as np
import yaml
from scipy.io import wavfile

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

RARE_DELIMITER = b"\xe2\x96\x81".decode("utf-8")  # A rare used delimiter as "▁"


# ==================== data utils ====================
def read_lines(data_path, log=True):
  """ read from txt file, and return all lines as list, and
  remove \t or \n in end of every line

  Args:
    data_path: txt path
    log: True to output log

  Returns:
    List

  """
  lines = []
  with open(data_path, encoding="utf-8") as fr:
    for line in fr.readlines():
      if len(line.strip().replace(" ", "")):
        lines.append(line.strip())
  if log:
    print("read {} lines from {}".format(len(lines), data_path))
    print("example(last) {}\n".format(lines[-1]))
  return lines


def write_lines(data_path, lines, log=True):
  """ write lines to txt path.

  Args:
    data_path: txt path
    lines: List
    log: True for output logs

  Returns:

  """

  if log:
    print("write {} lines to {}".format(len(lines), data_path))
    print("example(last line): {}\n".format(lines[-1]))

  if len(lines) == 0:
    raise Exception("Fails to write blank to {}".format(data_path))

  with open(data_path, "w", encoding="utf-8") as fw:
    for line in lines:
      fw.write("{}\n".format(line))
  return


def line_to_dict(line):
  """ change line to dict.
  line: "key1:value1\t key2:value2\t ..."

  Args:
    line: str

  Returns:
    Dict

  """
  if line.startswith("{"):
    return json.loads(line)
  info = line.strip().split("\t")
  data_dict = {}
  for data in info:
    name = data.split(":")[0]
    value = ":".join(data.split(":")[1:])
    data_dict[name] = value
  if "len" in data_dict.keys():
    data_dict["len"] = int(data_dict["len"])
  return data_dict


def dict_to_line(data_dict):
  """ change dict to line, inverse of function "line_to_dict"

  Args:
    data_dict: Dict

  Returns:
    List

  """
  line = json.dumps(data_dict, ensure_ascii=False)
  return line


def read_bin_to_numpy(data_path, data_type="float"):
  """ read matrix from bin file to numpy

  Args:
    data_path: bin file path
    data_type: type, must be "int" or "float"

  Returns:
    numpy

  """
  with open(data_path) as fr:
    frms = np.fromfile(fr, dtype=np.int32, count=1)[0]
    dim = np.fromfile(fr, dtype=np.int32, count=1)[0]
    if data_type == "float":
      mat = np.fromfile(fr, dtype=np.float32)
    elif data_type == "int":
      mat = np.fromfile(fr, dtype=np.int32)
    else:
      raise Exception("Unvalid type")

  assert np.size(mat) == frms * dim, "{}!={}x{}\ndata path:{}".format(
    np.size(mat), frms, dim, data_path)
  return np.reshape(mat, [frms, dim])


def dump_numpy_to_bin(data, data_path, data_type="float"):
  """ dump numpy to bin file, and data will be changed to float32

  Args:
    data: numpy
    data_path: bin path
    data_type: type must be "int" or "float"

  Returns:

  """
  if data_type == "float":
    data = data.astype(np.float32)
  elif data_type == "int":
    data = data.astype(np.int32)
  else:
    raise Exception("Unvalid type")

  if np.ndim(data) == 1:
    data = np.resize(data, [1, np.size(data)])
  m, n = np.shape(data)
  bin_shape = np.array([m, n], dtype=np.int32)
  with open(data_path, "wb") as fw:
    bin_shape.tofile(fw)
    data.tofile(fw)
  return


def load_map(map_path, key_as_int=False, value_as_int=False, split_label=" "):
  """ load map from file, file format maybe
  a 1
  b 2

  Args:
    map_path:
    key_as_int:
    value_as_int:

  Returns:

  """
  map_dict = {}
  for line in read_lines(map_path, log=False):
    assert len(line.split(split_label)) >= 2, "line:{}".format(line)
    if len(line.split(split_label)) == 2:
      k, v = line.split(split_label)
    else:
      v = line.split(split_label)[-1]
      k = " ".join(line.split(split_label)[:-1])

    if key_as_int:
      k = int(k)
    if value_as_int:
      v = int(v)
    map_dict[k] = v
  return map_dict


# ==================== file utils ====================
def clean_and_new_dir(data_dir):
  """ clean all files in dir

  Args:
    data_dir: str

  Returns:

  """
  if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
  os.makedirs(data_dir)
  return


def generate_dir_tree(dir_name, sub_name_list, del_old=False):
  """ generate new folder with sub folder

  Args:
    dir_name: father dir
    sub_name_list: sub dir names
    del_old: True to del old files

  Returns:

  """
  os.makedirs(dir_name, exist_ok=True)
  dir_path_list = []
  if del_old:
    shutil.rmtree(dir_name, ignore_errors=True)
  for name in sub_name_list:
    dir_path = os.path.join(dir_name, name)
    dir_path_list.append(dir_path)
    os.makedirs(dir_path, exist_ok=True)
  return dir_path_list


def get_name_from_path(abs_path):
  """ get name from path. eg: /xxx/yyy/zzz/name.prefix -> name

  Args:
    abs_path: str

  Returns:
    str

  """
  return ".".join(os.path.basename(abs_path).split(".")[:-1])


def get_wav_path_from_dir(wav_dir):
  """get all wav path in the input-dir"""
  path_list = []
  for root, dirs, files in os.walk(wav_dir):
    for name in files:
      wav_path = os.path.join(root, name)
      if wav_path.endswith("mp3") or wav_path.endswith("wav") or \
          wav_path.endswith("mp4"):
        path_list.append(wav_path)
  return sorted(path_list)


def remake_path_for_linux(path):
  """change file path for linux"""
  return path.replace(' ', '\ ').replace('(', '\(').replace(')', '\)').replace(
    '&', '\&').replace(";", "\;").replace("\'", "\\\'")


def create_rank_logger(local_rank=-1):
  """create logger with rank(especially for distribution train) """
  logger = logging.getLogger("RANK:{}".format(local_rank))
  ch = logging.StreamHandler()
  ch.setLevel(logging.INFO)
  if local_rank >= 0:
    ch.setFormatter(
      logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s'))
  else:
    ch.setFormatter(
      logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
  logger.addHandler(ch)
  logger.propagate = False
  logger.info("create logger of rank:{}".format(local_rank))
  return logger


# ==================== hparams utils ====================
def load_hparams(yaml_path):
  """ load hparams from yaml path as dict"""
  with open(yaml_path, encoding="utf-8") as yaml_file:
    hparams = yaml.safe_load(yaml_file)
  return hparams


def dump_hparams(yaml_path, hparams):
  """ dump dict to yaml path

  Args:
    yaml_path: str
    hparams: dict

  Returns:

  """
  with open(yaml_path, "w") as fw:
    yaml.dump(hparams, fw)
  print("save hparams to {}".format(yaml_path))
  return


def get_hparams_as_string(hparams):
  """ get hparams as format string

  Args:
    hparams: dict

  Returns:
    string

  """
  json_str = json.dumps(hparams, indent=2, separators=(',', ': '))
  return json_str


# ==================== wave utils ====================
def save_wav(wav, path, sr, k=None):
  if k:
    norm_wav = wav * 32767 / max(0.01, np.max(np.abs(wav))) * k
  else:
    norm_wav = wav * 32767
  wavfile.write(path, sr, norm_wav.astype(np.int16))
  return


if __name__ == '__main__':
  pass
