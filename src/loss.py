#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# author:liufeng
# datetime:2022/7/15 12:36 PM
# software: PyCharm

import logging
from typing import List
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import check_argument_types


class CenterLoss(nn.Module):
  """Center loss.

  Reference:
    A Discriminative Feature Learning Approach for Deep Face Recognition.

  Args:
      num_classes (int): number of classes.
      feat_dim (int): feature dimension.
  """

  def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
    super(CenterLoss, self).__init__()
    self.num_classes = num_classes
    self.feat_dim = feat_dim
    self.use_gpu = use_gpu
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    if self.use_gpu:
      self.centers = nn.Parameter(
        torch.randn(self.num_classes, self.feat_dim).cuda())
      self.centers = nn.Parameter(
        torch.randn(self.num_classes, self.feat_dim).to("cuda"))
    else:
      self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
    logging.info("CenterLoss: num_classes({}), feat_dim({})".format(
      num_classes, feat_dim))
    return

  def forward(self, x, labels):
    """
    Args:
        x: feature matrix with shape (batch_size, feat_dim).
        labels: ground truth labels with shape (batch_size).
    """
    batch_size = x.size(0)
    distmat = torch.pow(x, 2).sum(dim=1, keepdim=True). \
                expand(batch_size, self.num_classes) + \
              torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
                self.num_classes, batch_size).t()
    distmat.addmm_(1, -2, x, self.centers.t())

    classes = torch.arange(self.num_classes).long()
    if self.use_gpu:
      classes = classes.cuda()
    labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
    mask = labels.eq(classes.expand(batch_size, self.num_classes))

    dist = distmat * mask.float()
    loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

    return loss


class FocalLoss(nn.Module):
  """ Focal Loss implement for https://arxiv.org/abs/1708.02002 """

  def __init__(self, gamma: float = 2., alpha: List = None, num_cls: int = -1,
               reduction: str = 'mean'):

    super(FocalLoss, self).__init__()
    assert check_argument_types()
    if reduction not in ['mean', 'sum']:
      raise NotImplementedError(
        'Reduction {} not implemented.'.format(reduction))
    self._reduction = reduction
    self._alpha = alpha
    self._gamma = gamma
    if alpha is not None:
      assert len(alpha) <= num_cls, "{} != {}".format(len(alpha), num_cls)
      self._alpha = torch.tensor(self._alpha)
    self._eps = np.finfo(float).eps
    logging.info("Init Focal loss with gamma:{}".format(gamma))
    return

  def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """ compute focal loss for pred and label

    Args:
      y_pred: [batch_size, num_cls]
      y_true: [batch_size]

    Returns:
      loss
    """
    b = y_pred.size(0)
    y_pred_softmax = torch.nn.Softmax(dim=1)(y_pred) + self._eps
    ce = -torch.log(y_pred_softmax)
    ce = ce.gather(1, y_true.view(-1, 1))

    y_pred_softmax = y_pred_softmax.gather(1, y_true.view(-1, 1))
    weight = torch.pow(torch.sub(1., y_pred_softmax), self._gamma)

    if self._alpha is not None:
      self._alpha = self._alpha.to(y_pred.device)
      alpha = self._alpha.gather(0, y_true.view(-1))
      alpha = alpha.unsqueeze(1)
      alpha = alpha / torch.sum(alpha) * b
      weight = torch.mul(alpha, weight)
    fl_loss = torch.mul(weight, ce).squeeze(1)
    return self._reduce(fl_loss)

  def forward_with_onehot(self, y_pred: torch.Tensor,
                          y_true: torch.Tensor) -> torch.Tensor:
    """ Another implement for "forward" with onehot label matrix.

    It is not good because when ce_embed size is large, more memory will used.

    Args:
      y_pred: [batch_size, num_cls]
      y_true: [batch_size]

    Returns:
      loss

    """
    y_pred = torch.nn.Softmax(dim=1)(y_pred)
    y_true = torch.nn.functional.one_hot(y_true, self.num_cls)

    eps = np.finfo(float).eps
    y_pred = y_pred + eps
    ce = torch.mul(y_true, -torch.log(y_pred))
    weight = torch.mul(y_true, torch.pow(torch.sub(1., y_pred), self._gamma))
    fl_loss = torch.mul(self.alpha, torch.mul(weight, ce))

    fl_loss, _ = torch.max(fl_loss, dim=1)
    return self._reduce(fl_loss)

  def _reduce(self, x):
    if self._reduction == 'mean':
      return torch.mean(x)
    else:
      return torch.sum(x)


class HardTripletLoss(nn.Module):
  """Hard/Hardest Triplet Loss
    (pytorch implementation of https://omoindrot.github.io/triplet-loss)
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    """

  def __init__(self, margin=0.1):
    """ Args:
      margin: margin for triplet loss
    """
    super(HardTripletLoss, self).__init__()
    self._margin = margin
    return

  def forward(self, embeddings, labels):
    """
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    pairwise_dist = self._pairwise_distance(embeddings, squared=False)

    mask_anchor_positive = self._get_anchor_positive_triplet_mask(
      labels).float()
    valid_positive_dist = pairwise_dist * mask_anchor_positive
    hardest_positive_dist, _ = torch.max(valid_positive_dist, dim=1,
                                         keepdim=True)

    # Get the hardest negative pairs
    mask_negative = self._get_anchor_negative_triplet_mask(labels).float()
    max_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
    negative_dist = pairwise_dist + max_negative_dist * (1.0 - mask_negative)
    hardest_negative_dist, _ = torch.min(negative_dist, dim=1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = F.relu(
      hardest_positive_dist - hardest_negative_dist + self._margin)
    triplet_loss = torch.mean(triplet_loss)
    return triplet_loss

  @staticmethod
  def _pairwise_distance(x, squared=False, eps=1e-16):
    # Compute the 2D matrix of distances between all the embeddings.

    cor_mat = torch.matmul(x, x.t())
    norm_mat = cor_mat.diag()
    distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
    distances = F.relu(distances)

    if not squared:
      mask = torch.eq(distances, 0.0).float()
      distances = distances + mask * eps
      distances = torch.sqrt(distances)
      distances = distances * (1.0 - mask)
    return distances

  @staticmethod
  def _get_anchor_positive_triplet_mask(labels):
    """Return a 2D mask where mask[a, p] is True, if a and p are distinct and
       have same label.

    """
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = labels.device
    indices_not_equal = torch.eye(labels.shape[0]).to(device).byte() ^ 1
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = indices_not_equal * labels_equal
    return mask

  @staticmethod
  def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    """
    labels_equal = torch.unsqueeze(labels, 0) == torch.unsqueeze(labels, 1)
    mask = labels_equal ^ 1
    return mask
