#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2023/1/13 11:07 AM
# software: PyCharm

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.map import calc_map
from src.dataset import AudioFeatDataset
from src.utils import line_to_dict, dict_to_line
from src.utils import read_lines, write_lines, RARE_DELIMITER


def _calc_embed(model, query_loader, device, saved_dir=None):
  """ calculate audio embedding

  Args:
    model: trained model
    query_loader: dataset loader for audio
    device: cuda or cpu
    saved_dir: To save embed to disk as npy

  Returns:
    query_utt_label: List[(utt, label), ...],
    query_embed: Dict, key is utt, value is List with embed of every chunk

  """
  query_label = {}
  query_embed = {}
  with torch.no_grad():

    for j, batch in enumerate(query_loader):
      utt_b, feat_b, label_b = batch
      feat_b = torch.autograd.Variable(
        feat_b.to(device, non_blocking=True)).float()
      label_b = torch.autograd.Variable(
        label_b.to(device, non_blocking=True)).long()
      if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        embed_b, _ = model.module.inference(feat_b)
      else:
        embed_b, _ = model.inference(feat_b)

      embed_b = embed_b.cpu().numpy()
      label_b = label_b.cpu().numpy()
      for idx_embed in range(len(embed_b)):
        utt = utt_b[idx_embed]
        embed = embed_b[idx_embed]
        label = label_b[idx_embed]

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
          assert np.shape(embed) == (
            model.module.get_embed_length(),), np.shape(embed)
        else:
          assert np.shape(embed) == (model.get_embed_length(),), np.shape(embed)

        if utt not in query_label.keys():
          query_label[utt] = label
        else:
          assert query_label[utt] == label

        if utt not in query_embed.keys():
          query_embed[utt] = []
        query_embed[utt].append(embed)
        if saved_dir:
          saved_path = os.path.join(saved_dir, "{}.npy".format(utt))
          np.save(saved_path, embed)
  query_utt_label = sorted(list(query_label.items()))
  return query_utt_label, query_embed


def _generate_dist_matrix(query_utt_label, query_embed, ref_utt_label=None,
                          ref_embed=None, query_in_ref=None):
  """generate distance matrix from query/ref embedding

  Args:
    query_utt_label: List[(utt, label), ...],
    query_embed: Dict, key is utt, value is List with embed of every chunk
    ref_utt_label: List[(utt, label), ...]
    ref_embed: Dict, key is utt, value is List with embed of every chunk
    query_in_ref: List[(idx, idy), ...], means query[idx] is in ref[idy] so
                  we skip that when computing map

  Returns:
    dist_matrix: [numpy.ndarray]
    query_label:
    ref_label:

  """
  if ref_utt_label is None and ref_embed is None:
    query_in_ref = [(i, i) for i in range(len(query_utt_label))]
    ref_utt_label = query_utt_label
    ref_embed = query_embed

  dist_matrix = np.zeros([len(query_utt_label), len(ref_utt_label)])
  for idx, (utt_query, _) in enumerate(query_utt_label):
    for idy, (utt_ref, _) in enumerate(ref_utt_label):
      to_choice = []
      for embed_x in query_embed[utt_query]:
        for embed_y in ref_embed[utt_ref]:
          embed_x = embed_x / np.linalg.norm(embed_x)  # todo:
          embed_y = embed_y / np.linalg.norm(embed_y)  # todo:
          cos_sim = embed_x.dot(embed_y)
          dist = 1 - cos_sim
          to_choice.append(dist)
      dist_matrix[idx, idy] = min(to_choice)

  if query_in_ref:
    for idx, idy in query_in_ref:
      dist_matrix[idx, idy] = -1  # will be skip when compute map

  query_label = [v for k, v in query_utt_label]
  ref_label = [v for k, v in ref_utt_label]
  return dist_matrix, query_label, ref_label


def _load_chunk_embed_from_dir(query_chunk_lines):
  query_utt_label = []
  query_embed = {}
  utt_s = set()
  for line in query_chunk_lines:
    local_data = line_to_dict(line)
    utt = local_data["utt"].split("-{}start-".format(RARE_DELIMITER))[0]
    label = local_data["song_id"]
    if utt not in utt_s:
      query_utt_label.append((utt, label))
      utt_s.add(utt)
      query_embed[utt] = []
    query_embed[utt].append(np.load(local_data["embed"]))
  return query_utt_label, query_embed


def _cut_one_line_with_dur(line, window_length_s, window_shift_s,
                           hop_size=0.04):
  """cut line with window_length_s

  If duration is smaller than window_length_s, return only one chunk.

  Args:
    line:
    window_length_s:
    window_shift_s:

  Returns:
    lines with "start" info

  """
  local_data = line_to_dict(line)
  utt = local_data["utt"]
  if "dur_s" in local_data.keys():
    dur_s = local_data["dur_s"]
  else:
    dur_s = local_data["dur_ms"] / 1000

  short_lines = []
  start_s = 0.0
  while start_s + window_length_s < dur_s or start_s == 0:
    local_data["start_s"] = start_s
    local_data["start"] = int(start_s * 25)
    local_data["utt"] = "{}-{}start-{}".format(utt, RARE_DELIMITER,
                                               int(start_s))
    short_lines.append(dict_to_line(local_data))
    start_s += window_shift_s
  return short_lines


def _cut_lines_with_dur(init_lines, chunk_s, embed_dir):
  """cut line with duration: for every line, add "start" for signal start point

  Args:
    init_lines: list of all lines
    chunk_s: float, second for every chunk
    embed_dir: embed dir to save embed array

  Returns:
    chunk_lines: same with init_line beside "start"
  """
  os.makedirs(embed_dir, exist_ok=True)
  chunk_lines = []
  for line in init_lines:
    short_lines = _cut_one_line_with_dur(line, window_length_s=chunk_s,
                                         window_shift_s=chunk_s)
    for short_line in short_lines:
      local_data = line_to_dict(short_line)
      local_data["embed"] = os.path.join(
        embed_dir, "{}.npy".format(local_data["utt"]))
      chunk_lines.append(dict_to_line(local_data))
  return chunk_lines


def eval_for_map_with_feat(hp, model, embed_dir, query_path, ref_path,
                           query_in_ref_path=None, batch_size=128,
                           num_workers=1,
                           device="cuda", logger=None):
  """compute map10 with trained model and query/ref loader(dataset loader
  can speed up process dramatically)

  Args:
    num_workers:
    hp: dict contains hparams
    model: nnet model, should has method 'infer'
    embed_dir: dir for saving embedding, None for not saving anything
    query_path: contains query info
    ref_path: contains ref info
    query_in_ref_path: path to store query in ref index, None means that
        query index equals ref index
    batch_size: for nnet infer
    device: "cuda" or "cpu"
    logger:

  Returns:
    map10
    rank1

  """
  if logger:
    logger.info("=" * 40)
    logger.info("Start to Eval")
    logger.info("query_path: {}".format(query_path))
    logger.info("ref_path: {}".format(ref_path))
    logger.info("query_in_ref_path: {}".format(query_in_ref_path))
    logger.info("using batch-size: {}".format(batch_size))

  os.makedirs(embed_dir, exist_ok=True)

  model.eval()
  model = model.to(device)

  if isinstance(hp["chunk_frame"], list):
    infer_frame = hp["chunk_frame"][0] * hp["mean_size"]
  else:
    infer_frame = hp["chunk_frame"] * hp["mean_size"]

  chunk_s = hp["chunk_s"]
  assert infer_frame == chunk_s * 25, \
    "Error for mismatch of chunk_frame and chunk_s: {}!={}*25".format(
      infer_frame, chunk_s)

  query_lines = read_lines(query_path, log=False)
  ref_lines = read_lines(ref_path, log=False)
  if logger:
    logger.info("query lines: {}".format(len(query_lines)))
    logger.info("ref lines: {}".format(len(ref_lines)))
    logger.info("chunk_frame: {} chunk_s:{}\n".format(infer_frame, chunk_s))

  if query_in_ref_path:
    line = read_lines(query_in_ref_path, log=False)[0]
    query_in_ref = line_to_dict(line)["query_in_ref"]
    for idx, idy in query_in_ref:
      assert idx < len(query_lines), \
        "query idx {} must be smaller that max query idx {}".format(
          idx, len(query_lines))
      assert idy < len(ref_lines), \
        "ref idx {} must be smaller that max ref idx {}".format(
          idy, len(ref_lines))
  else:
    query_in_ref = None

  query_embed_dir = os.path.join(embed_dir, "query_embed")
  query_chunk_lines = _cut_lines_with_dur(query_lines, chunk_s, query_embed_dir)
  write_lines(os.path.join(embed_dir, "query.txt"), query_chunk_lines, False)
  to_cal_lines = [l for l in query_chunk_lines
                  if not os.path.exists(line_to_dict(l)["embed"])]
  if logger:
    logger.info("query chunk lines: {}, to compute lines: {}".format(
      len(query_chunk_lines), len(to_cal_lines)))

  if len(to_cal_lines) > 0:
    data_loader = DataLoader(AudioFeatDataset(hp, data_lines=to_cal_lines,
                                              mode="defined",
                                              chunk_len=infer_frame),
                             num_workers=num_workers,
                             shuffle=False,
                             sampler=None,
                             batch_size=batch_size,
                             pin_memory=True,
                             collate_fn=None,
                             drop_last=False)
    _calc_embed(model, data_loader, device, saved_dir=query_embed_dir)

  ref_embed_dir = os.path.join(embed_dir, "query_embed")
  ref_chunk_lines = _cut_lines_with_dur(ref_lines, chunk_s, ref_embed_dir)
  write_lines(os.path.join(embed_dir, "ref.txt"), ref_chunk_lines, False)
  if ref_path != query_path:
    to_cal_lines = [l for l in ref_chunk_lines
                    if not os.path.exists(line_to_dict(l)["embed"])]
    if logger:
      logger.info("ref chunk lines: {}, to compute lines: {}".format(
        len(ref_chunk_lines), len(to_cal_lines)))
    if len(to_cal_lines) > 0:
      data_loader = DataLoader(AudioFeatDataset(hp, data_lines=to_cal_lines,
                                                mode="defined",
                                                chunk_len=infer_frame),
                               num_workers=num_workers,
                               shuffle=True,
                               sampler=None,
                               batch_size=batch_size,
                               pin_memory=True,
                               collate_fn=None,
                               drop_last=False)
      _calc_embed(model, data_loader, device, saved_dir=ref_embed_dir)
    if logger:
      logger.info(
        "Finish computing ref embedding, saved at {}\n".format(ref_embed_dir))
  else:
    if logger:
      logger.info("Because query and ref have same path, "
                  "so skip to compute ref embedding")

  query_utt_label, query_embed = _load_chunk_embed_from_dir(query_chunk_lines)
  if ref_path == query_path:
    ref_utt_label, ref_embed = None, None
  else:
    ref_utt_label, ref_embed = _load_chunk_embed_from_dir(ref_chunk_lines)
  if logger:
    logger.info("Finish loading embedding and Start to compute dist matrix")

  dist_matrix, query_label, ref_label = _generate_dist_matrix(
    query_utt_label, query_embed, ref_utt_label, ref_embed,
    query_in_ref=query_in_ref)

  if logger:
    logger.info("Finish computing distance matrix and Start to compute map")
    logger.info(
      "Inp dist shape: {}, query: {}, ref: {}".format(np.shape(dist_matrix),
                                                      len(query_label),
                                                      len(ref_label)))

  metrics = calc_map(dist_matrix, query_label, ref_label,
                     topk=10000, verbose=0)
  if logger:
    logger.info("map: {}".format(metrics["mean_ap"]))
    logger.info("rank1: {}".format(metrics["rank1"]))
    logger.info("hit_rate: {}\n".format(metrics["hit_rate"]))
  return metrics["mean_ap"], metrics["hit_rate"],  metrics["rank1"]


if __name__ == '__main__':
  pass
