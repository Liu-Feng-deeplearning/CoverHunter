#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import logging
import random
from contextlib import nullcontext

import numpy as np
import torch

from src.pytorch_utils import scan_and_load_checkpoint, get_lr

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


