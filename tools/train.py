#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import argparse
import os
import time
from argparse import RawTextHelpFormatter

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.scheduler import UserDefineExponentialLR
from src.dataset import AudioFeatDataset, MPerClassSampler
from src.trainer import save_checkpoint, load_checkpoint
from src.trainer import train_one_epoch, validate
from src.eval_testset import eval_for_map_with_feat
from src.utils import load_hparams, get_hparams_as_string, create_rank_logger
from src.model import Model

torch.backends.cudnn.benchmark = True


def _main():
  """support distribution(ddp)"""
  parser = argparse.ArgumentParser(
    description="Train: python3 -m tools.train model_dir\n\n"
                "Train for ddp: \n"
                "torchrun -m --nnodes=1 --nproc_per_node=2 tools.train "
                "model_dir\n",
    formatter_class=RawTextHelpFormatter)
  parser.add_argument('model_dir')
  parser.add_argument('--first_eval', default=False, action='store_true',
                      help="Set for run eval first before train")
  parser.add_argument('--only_eval', default=False, action='store_true',
                      help="Set for run eval first before train")
  parser.add_argument('--debug', default=False, action='store_true',
                      help="give more debug log")
  args = parser.parse_args()
  model_dir = args.model_dir
  first_eval = args.first_eval
  only_eval = args.only_eval
  first_eval = True if only_eval else first_eval
  assert torch.cuda.is_available()

  local_rank = int(os.environ.get("LOCAL_RANK", -1))
  if local_rank >= 0:
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.distributed.init_process_group(backend='nccl')
    total_rank = torch.distributed.get_world_size()
  else:
    device = torch.device('cuda:0')
    total_rank = -1

  logger = create_rank_logger(local_rank)
  logger.info("local rank-{}, total rank-{}".format(local_rank, total_rank))

  hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
  if local_rank <= 0:
    logger.info("{}".format(get_hparams_as_string(hp)))

  torch.manual_seed(hp["seed"])

  # We use multi length sample to train
  train_loader_lst = []
  for chunk_len in hp["chunk_frame"]:
    train_dataset = AudioFeatDataset(
      hp, hp["train_path"], train=True, mode=hp["mode"],
      chunk_len=chunk_len * hp["mean_size"],
      logger=(logger if local_rank <= 0 else None))
    sampler = MPerClassSampler(data_path=hp["train_path"],
                               m=hp["m_per_class"],
                               batch_size=hp["batch_size"],
                               distribute=(local_rank >= 0), logger=logger)
    train_loader = DataLoader(
      train_dataset,
      num_workers=hp["num_workers"],
      shuffle=(sampler is None),
      sampler=sampler,
      batch_size=hp["batch_size"],
      pin_memory=True,
      drop_last=True,
    )
    train_loader_lst.append(train_loader)

  # At inference stage, we only use chunk with fixed length
  if local_rank <= 0:
    logger.info("Init train-sample and dev data loader")
  infer_len = hp["chunk_frame"][0] * hp["mean_size"]
  if "train_sample_path" in hp.keys():
    # hp["batch_size"] = 1
    dataset = AudioFeatDataset(
      hp, hp["train_sample_path"], train=False, chunk_len=infer_len,
      mode=hp["mode"], logger=(logger if local_rank <= 0 else None))
    sampler = MPerClassSampler(data_path=hp["train_sample_path"],
                               # m=hp["m_per_class"],
                               m=1,
                               batch_size=hp["batch_size"],
                               distribute=(local_rank >= 0), logger=logger)
    train_sampler_loader = DataLoader(
      dataset,
      num_workers=1,
      shuffle=False,
      sampler=sampler,
      batch_size=hp["batch_size"],
      pin_memory=True,
      collate_fn=None,
      drop_last=False)
  else:
    train_sampler_loader = None

  if "dev_path" in hp.keys():
    dataset = AudioFeatDataset(hp, hp["dev_path"], chunk_len=infer_len,
                               mode=hp["mode"],
                               logger=(logger if local_rank <= 0 else None))
    sampler = MPerClassSampler(data_path=hp["dev_path"],
                               m=hp["m_per_class"],
                               batch_size=hp["batch_size"],
                               distribute=(local_rank >= 0), logger=logger)
    dev_loader = DataLoader(
      dataset,
      num_workers=1,
      shuffle=False,
      sampler=sampler,
      batch_size=hp["batch_size"],
      pin_memory=True,
      collate_fn=None,
      drop_last=False)
  else:
    dev_loader = None

  # we use map-reduce mode to update model when its parameters changed
  # (model.join), that means we do not need to wait one step of all gpu to
  # complete. Pytorch distribution support variable trained samples of different
  # gpus.
  # And also, we compute train-sample/dev/testset on different gpu within epoch.
  # For example: we compute dev at rank0 when epoch 1, when dev is computing,
  # rank1 is going on training and update parameters. When epoch 2, we change
  # to compute dev at rank1, to make sure all ranks run the same train steps
  # almost.
  all_test_set_list = ["covers80", "shs_test", "dacaos", "hymf_20", "hymf_100"]
  test_set_list = [d for d in all_test_set_list if d in hp.keys()]

  model = Model(hp).to(device)
  checkpoint_dir = os.path.join(model_dir, "pt_model")
  os.makedirs(checkpoint_dir, exist_ok=True)

  optimizer = torch.optim.AdamW(
    model.parameters(), hp["learning_rate"],
    betas=[hp["adam_b1"], hp["adam_b2"]])
  step, init_epoch = load_checkpoint(model, optimizer, checkpoint_dir,
                                     advanced=False)

  if local_rank >= 0:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
      model, device_ids=[local_rank], output_device=local_rank)

  scheduler = UserDefineExponentialLR(
    optimizer, gamma=hp["lr_decay"], min_lr=hp["min_lr"],
    last_epoch=init_epoch)

  sw = SummaryWriter(os.path.join(model_dir, "logs"))
  if only_eval:
    sw = None

  for epoch in range(max(0, 1 + init_epoch), 100000):
    if not first_eval:
      start = time.time()
      if local_rank >= 0:
        item_length = float("inf")
        for i, loader in enumerate(train_loader_lst):
          loader.sampler.shuffle_data_on_ranks(
            seed=epoch + i, display_details=(i == 0))
          length = loader.sampler.num_iters()
          item_length = length if length < item_length else item_length

        # logger.info("{}".format(item_length))
        train_step_t = torch.tensor([item_length], dtype=torch.long,
                                    device=device)
        torch.distributed.all_reduce(train_step_t,
                                     torch.distributed.ReduceOp.MIN,
                                     async_op=False)
        train_step = train_step_t.item() * len(train_loader_lst)
      else:
        train_step = None

      if local_rank >= 0:
        torch.distributed.barrier()
      logger.info("Start to train for epoch {}".format(epoch))
      step = train_one_epoch(model, optimizer, scheduler, train_loader_lst,
                             step, train_step=train_step,
                             sw=(sw if local_rank <= 0 else None),
                             logger=(logger if local_rank <= 0 else None))
      if local_rank >= 0:
        torch.distributed.barrier()

      if local_rank <= 0:
        if epoch % hp["every_n_epoch_to_save"] == 0:
          save_checkpoint(model, optimizer, step, epoch, checkpoint_dir)
        logger.info('Time for train epoch {} step {} is {:.1f}s\n'.format(
          epoch, step, time.time() - start))

    if train_sampler_loader and epoch % hp["every_n_epoch_to_dev"] == 0:
      if local_rank >= 0:
        torch.distributed.barrier()
      start = time.time()
      if local_rank <= 0:
        logger.info("compute train-sample at epoch-{} with rank {}".format(
          epoch, local_rank))

      res = validate(model, train_sampler_loader, "train_sample",
                     epoch_num=epoch,
                     sw=(sw if local_rank <= 0 else None),
                     logger=(logger if local_rank <= 0 else None))
      logger.info(
        "count:{}, avg_ce_loss:{}".format(
          res["count"], res["ce_loss"] / res["count"]))

      if local_rank >= 0:
        total = torch.tensor([res["count"], res["ce_loss"]],
                             dtype=torch.float32, device=device)
        torch.distributed.all_reduce(total, torch.distributed.ReduceOp.SUM,
                                     async_op=False)
        tot_count, tot_ce_loss = total.tolist()
        avg = tot_ce_loss / tot_count
        if local_rank == 0:
          logger.info(
            "Train sample count:{}, avg_ce_loss:{:.4f}".format(tot_count, avg))
          sw.add_scalar("csi_{}/{}".format("train_sample", "avg_ce_loss"),
                        avg, epoch)

      if local_rank <= 0:
        logger.info('Time for train-sample is {:.1f}s\n'.format(
          time.time() - start))

    if dev_loader and epoch % hp["every_n_epoch_to_dev"] == 0:
      start = time.time()
      if local_rank <= 0:
        logger.info(
          "compute dev at epoch-{} with rank {}".format(epoch, local_rank))
      dev_res = validate(model, dev_loader, "dev", epoch_num=epoch,
                         sw=(sw if local_rank <= 0 else None),
                         logger=(logger if local_rank <= 0 else None))
      cnt = dev_res["count"]
      logger.info(
        "count:{}, avg_ce_loss:{}".format(cnt, dev_res["ce_loss"] / cnt))
      if local_rank >= 0:
        total = torch.tensor([cnt, dev_res["ce_loss"]],
                             dtype=torch.float32, device=device)
        torch.distributed.all_reduce(total, torch.distributed.ReduceOp.SUM,
                                     async_op=False)
        tot_count, tot_ce_loss = total.tolist()
        avg = tot_ce_loss / tot_count
        if local_rank == 0:
          logger.info(
            "Avg of rank:: count:{}, avg_loss:{}".format(tot_count, avg))
          sw.add_scalar("csi_{}/{}".format("dev", "avg_ce_loss"), avg, epoch)

      if local_rank <= 0:
        logger.info('Time for dev is {:.1f}s\n'.format(time.time() - start))

    valid_testlist = []
    for test_idx, testset_name in enumerate(test_set_list):
      hp_test = hp[testset_name]
      if epoch % hp_test["every_n_epoch_to_dev"] == 0:
        valid_testlist.append(testset_name)

    for test_idx, testset_name in enumerate(valid_testlist):
      hp_test = hp[testset_name]
      if test_idx % total_rank == local_rank or local_rank == -1:
        logger.info(
          "Compute {} with rank {} at epoch: {}".format(testset_name,
                                                        local_rank, epoch))

        start = time.time()
        save_name = hp_test.get("save_name", testset_name)
        embed_dir = os.path.join(model_dir,
                                 "embed_{}_{}".format(epoch, save_name))
        query_in_ref_path = hp_test.get("query_in_ref_path", None)
        mean_ap, hit_rate = eval_for_map_with_feat(
          hp, model, embed_dir, query_path=hp_test["query_path"],
          ref_path=hp_test["ref_path"], query_in_ref_path=query_in_ref_path,
          batch_size=hp["batch_size"], logger=logger)

        # eval_for_map_with_feat_stage1(
        #   hp, embed_dir, query_path=hp_test["query_path"],
        #   ref_path=hp_test["ref_path"], query_in_ref_path=query_in_ref_path,
        #   logger=logger)

        sw.add_scalar("mAP/{}".format(testset_name), mean_ap, epoch)
        sw.add_scalar("hit_rate/{}".format(testset_name), hit_rate, epoch)
        logger.info("Test {}, hit_rate:{}, map:{}".format(
          testset_name, hit_rate, mean_ap))
        logger.info('Time for test-{} is {} sec\n'.format(
          testset_name, int(time.time() - start)))
        # if local_rank >= 0:
        #   torch.distributed.barrier()
    if only_eval:
      return
    first_eval = False
  return


if __name__ == '__main__':
  _main()
