#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
from src.map import calc_map
from torch.utils.data import DataLoader

from src.dataset import AudioFeatDataset
from src.pytorch_utils import scan_and_load_checkpoint, get_lr
from src.utils import line_to_dict
from src.utils import read_lines
from src.utils import write_lines

torch.backends.cudnn.benchmark = True


def save_checkpoint(model, optimizer, step, epoch, checkpoint_dir):
  g_checkpoint_path = "{}/g_{:08d}".format(checkpoint_dir, epoch)

  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()

  torch.save({'generator': state_dict}, g_checkpoint_path)
  d_checkpoint_path = "{}/do_{:08d}".format(checkpoint_dir, epoch)
  torch.save({'optim_g': optimizer.state_dict(),
              'steps': step,
              'epoch': epoch}, d_checkpoint_path)
  logging.info("save checkpoint to {}".format(g_checkpoint_path))
  logging.info("save step:{}, epoch:{}".format(step, epoch))
  return


def load_checkpoint(model, optimizer=None, checkpoint_dir=None, advanced=False):
  state_dict_g = scan_and_load_checkpoint(checkpoint_dir, 'g_')
  state_dict_do = scan_and_load_checkpoint(checkpoint_dir, 'do_')
  if state_dict_g:

    if advanced:
      model_dict = model.state_dict()
      valid_dict = {k: v for k, v in state_dict_g.items() if
                    k in model_dict.keys()}
      model_dict.update(valid_dict)
      model.load_state_dict(model_dict)
      for k in model_dict.keys():
        if k not in state_dict_g.keys():
          logging.warning("{} not be initialized".format(k))
    else:
      model.load_state_dict(state_dict_g['generator'])
      # self.load_state_dict(state_dict_g)

    logging.info("load g-model from {}".format(checkpoint_dir))

  if state_dict_do is None:
    logging.info("using init value of steps and epoch")
    step, epoch = 1, -1
  else:
    step, epoch = state_dict_do['steps'] + 1, state_dict_do['epoch']
    logging.info("load d-model from {}".format(checkpoint_dir))
    optimizer.load_state_dict(state_dict_do['optim_g'])

  logging.info("step:{}, epoch:{}".format(step, epoch))
  return step, epoch



def train_one_epoch(model, optimizer, scheduler, train_loader_lst,
                    step, train_step=None, sw=None, logger=None):
  """train one epoch with multi data_loader, support distributed model"""
  if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model_context = model.join
  else:
    model_context = nullcontext
  # model_context = nullcontext
  init_step = step
  model.train()
  idx_loader = [i for i in range(len(train_loader_lst))]
  with model_context():
    for batch_lst in zip(*train_loader_lst):
      random.shuffle(idx_loader)
      for idx in idx_loader:
        batch = list(batch_lst)[idx]
        if step % 1000 == 0:
          scheduler.step()
        model.train()
        device = "cuda"
        _, feat, label = batch
        feat = torch.autograd.Variable(
          feat.to(device, non_blocking=True)).float()
        label = torch.autograd.Variable(
          label.to(device, non_blocking=True)).long()

        optimizer.zero_grad()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
          total_loss, losses = model.module.compute_loss(feat, label)
        else:
          total_loss, losses = model.compute_loss(feat, label)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        _loss_memory = {"lr": get_lr(optimizer)}
        for key, value in losses.items():
          _loss_memory.update({key: value.item()})
        _loss_memory.update({"total": total_loss.item()})

        if step % 100 == 0:
          log_info = "Steps:{:d}".format(step)
          for k, v in _loss_memory.items():
            if k == "lr":
              log_info += " lr:{:.6f}".format(v)
            else:
              log_info += " {}:{:.3f}".format(k, v)
            if sw:
              sw.add_scalar("csi/{}".format(k), v, step)
          if logger:
            logger.info(log_info)
        step += 1

        if train_step is not None:
          if (step - init_step) == train_step:
            return step
  return step


def validate(model, validation_loader, valid_name, sw=None, epoch_num=-1,
             device="cuda", logger=None):
  """ Validation on dataset, support distributed model"""
  model.eval()
  # model.module.eval()
  val_losses = {"count": 0}
  with torch.no_grad():
    for j, batch in enumerate(validation_loader):
      utt, anchor, label = batch
      anchor = torch.autograd.Variable(
        anchor.to(device, non_blocking=True)).float()
      label = torch.autograd.Variable(
        label.to(device, non_blocking=True)).long()
      if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        tot_loss, losses = model.module.compute_loss(anchor, label)
      else:
        tot_loss, losses = model.compute_loss(anchor, label)

      if logger and j % 10 == 0:
        logger.info("step-{} {} {} {} {}".format(
          j, utt[0], losses["ce_loss"].item(), anchor[0][0][0], label[0]))

      val_losses["count"] += 1
      for key, value in losses.items():
        if key not in val_losses.keys():
          val_losses[key] = 0.0
        val_losses[key] += losses[key].item()

    log_str = "{}: ".format(valid_name)
    for key, value in val_losses.items():
      if key == "count":
        continue
      value = value / (val_losses["count"])
      log_str = log_str + "{}-{:.3f} ".format(key, value)
      if sw is not None:
        sw.add_scalar("csi_{}/{}".format(valid_name, key), value, epoch_num)
  # if logger:
  #   logger.info(log_str)
  return val_losses


def _calc_label(model, query_loader, device):
  query_label = {}
  query_pred = {}
  with torch.no_grad():
    for j, batch in enumerate(query_loader):
      utt_b, anchor_b, label_b = batch
      anchor_b = torch.autograd.Variable(
        anchor_b.to(device, non_blocking=True)).float()
      label_b = torch.autograd.Variable(
        label_b.to(device, non_blocking=True)).long()
      _, pred_b = model.inference(anchor_b)
      pred_b = pred_b.cpu().numpy()
      label_b = label_b.cpu().numpy()

      for idx_embed in range(len(pred_b)):
        utt = utt_b[idx_embed]
        pred_embed = pred_b[idx_embed]
        pred_label = np.argmax(pred_embed)
        prob = pred_embed[pred_label]
        label = label_b[idx_embed]
        assert np.shape(pred_embed) == (model.get_ce_embed_length(),), \
          "invalid embed shape:{}".format(np.shape(pred_embed))
        if utt not in query_label.keys():
          query_label[utt] = label
        else:
          assert query_label[utt] == label

        if utt not in query_pred.keys():
          query_pred[utt] = []
        query_pred[utt].append((pred_label, prob))

  query_utt_label = sorted(list(query_label.items()))
  return query_utt_label, query_pred


def _syn_pred_label(model, valid_loader, valid_name, sw=None, epoch_num=-1,
                    device="cuda"):
  model.eval()

  query_utt_label, query_pred = _calc_label(model, valid_loader, device)

  utt_right, utt_total = 0, 0
  right, total = 0, 0
  for utt, label in query_utt_label:
    pred_lst = query_pred[utt]
    total += len(pred_lst)
    for pred, _ in pred_lst:
      right = right + 1 if pred == label else right

    utt_pred = sorted(pred_lst, key=lambda x: x[1], reverse=False)[0][0]
    utt_total += 1
    utt_right = utt_right + 1 if utt_pred == label else utt_right

  utt_acc = utt_right / utt_total
  acc = right / total
  if sw is not None:
    sw.add_scalar("coi_{}/utt_acc".format(valid_name), utt_acc, epoch_num)
    sw.add_scalar("coi_{}/acc".format(valid_name), acc, epoch_num)

  logging.info(
    "{} Utt Acc: {:.3f}, Total: {}".format(valid_name, utt_acc, utt_total))
  logging.info(
    "{} Acc: {:.3f}, Total: {}".format(valid_name, acc, total))
  return


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
  return metrics["mean_ap"], metrics["hit_rate"], metrics["rank1"]


