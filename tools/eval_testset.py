#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: liufeng
# datetime: 2023/3/16 3:41 PM


import os

import torch

from src.eval_testset import eval_for_map_with_feat
from src.model import Model
from src.utils import load_hparams, get_hparams_as_string, create_rank_logger
import argparse

torch.backends.cudnn.benchmark = True


def _main():
  parser = argparse.ArgumentParser(
    description="evaluate test-set with pretrained model")
  parser.add_argument('model_dir')
  parser.add_argument('query_path')
  parser.add_argument('ref_path')
  parser.add_argument('-query_in_ref_path', default="", type=str)

  args = parser.parse_args()
  model_dir = args.model_dir
  query_path = args.query_path
  ref_path = args.ref_path
  query_in_ref_path = args.query_in_ref_path

  assert torch.cuda.is_available()

  device = torch.device('cuda:0')
  total_rank = -1
  local_rank = -1

  logger = create_rank_logger(local_rank)
  logger.info("local rank-{}, total rank-{}".format(local_rank, total_rank))

  hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
  logger.info("{}".format(get_hparams_as_string(hp)))

  torch.manual_seed(hp["seed"])

  model = Model(hp).to(device)
  checkpoint_dir = os.path.join(model_dir, "pt_model")
  os.makedirs(checkpoint_dir, exist_ok=True)
  epoch = model.load_model_parameters(checkpoint_dir)

  embed_dir = os.path.join(model_dir, "embed_{}_{}".format(epoch, "tmp"))
  mean_ap, hit_rate, rank1 = eval_for_map_with_feat(
    hp, model, embed_dir, query_path=query_path,
    ref_path=ref_path, query_in_ref_path=query_in_ref_path,
    batch_size=64, logger=logger)

  logger.info("Test, map:{} rank1:{}".format(mean_ap, rank1))
  return


if __name__ == '__main__':
  _main()
