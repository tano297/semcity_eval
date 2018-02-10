#!/usr/bin/python3

import numpy as np


def calculate_confusion(mask, lbl, num_classes):
  '''
    get individual mask and label and create confusion matrix
  '''

  # flatten mask and cast
  flat_mask = np.array(mask).flatten().astype(np.uint32)
  # flatten label and cast
  flat_label = np.array(lbl).flatten().astype(np.uint32)
  # get the histogram
  confusion, _, _ = np.histogram2d(flat_label,
                                   flat_mask,
                                   bins=num_classes)
  return confusion


def pix_metrics_from_confusion(confusion):
  '''
    get complete confusion matrix and return:
      mean accuracy
      per class iou
      mean iou
      per class precision
      per class recall
  '''
  # calculate accuracy from confusion
  if confusion.sum():
    mean_acc = np.diag(confusion).sum() / confusion.sum()
  else:
    mean_acc = 0

  # calculate IoU
  per_class_iou = np.divide(np.diag(confusion), confusion.sum(
      0) + confusion.sum(1) - np.diag(confusion))
  mean_iou = np.nanmean(per_class_iou)

  # calculate precision and recall
  per_class_prec = np.divide(np.diag(confusion), confusion.sum(0))
  per_class_rec = np.divide(np.diag(confusion), confusion.sum(1))

  return mean_acc, mean_iou, per_class_iou, per_class_prec, per_class_rec
