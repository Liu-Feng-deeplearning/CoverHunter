#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author: liufeng
# datetime: 2023/7/5 6:08 PM


import os

import torch

from src.eval_testset import eval_for_map_with_feat
from src.model import Model
from src.utils import load_hparams, get_hparams_as_string, create_rank_logger
import argparse
from src.Aligner import Aligner

torch.backends.cudnn.benchmark = True


def _main():
  parser = argparse.ArgumentParser(
    description="alignment with coarse trained model")
  parser.add_argument('model_dir', help="coarse trained dir")
  parser.add_argument('data_path', help="input file contains init data")
  parser.add_argument('alignment_path',
                      help="output file contains alignment information")

  args = parser.parse_args()
  model_dir = args.model_dir
  data_path = args.data_path
  alignment_path = args.alignment_path

  assert torch.cuda.is_available()

  device = torch.device('cuda:0')
  total_rank = -1
  local_rank = -1

  logger = create_rank_logger(local_rank)
  logger.info("local rank-{}, total rank-{}".format(local_rank, total_rank))

  hp = load_hparams(os.path.join(model_dir, "config/hparams.yaml"))
  logger.info("{}".format(get_hparams_as_string(hp)))

  # Note: we need to change chunks to 15s
  hp["chunk_frame"] = [125]
  hp["chunk_s"] = 15  # = 125 / 25 * 3

  model = Model(hp).to(device)
  checkpoint_dir = os.path.join(model_dir, "pt_model")
  os.makedirs(checkpoint_dir, exist_ok=True)
  epoch = model.load_model_parameters(checkpoint_dir)

  # Calculate all chunks embedding and dump.
  embed_dir = os.path.join(model_dir, "embed_{}_{}".format(epoch, "tmp"))
  mean_ap, hit_rate, rank1 = eval_for_map_with_feat(
    hp, model, embed_dir, query_path=data_path,
    ref_path=data_path, query_in_ref_path=None,
    batch_size=64, logger=logger)
  logger.info("Test, map:{} rank1:{}".format(mean_ap, rank1))

  # Calculate shift frames for every-two items with same label
  aligner = Aligner(os.path.join(embed_dir, "query_embed"))
  aligner.align(data_path, alignment_path)
  logger.info("Output alignment into {}".format(alignment_path))
  return


if __name__ == '__main__':
  _main()
