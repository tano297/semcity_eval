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
      kappa coefficient
  '''
  conf = confusion.astype(np.float32)

  # calculate accuracy from confusion
  if conf.sum():
    mean_acc = np.diag(conf).sum() / conf.sum()
  else:
    mean_acc = 0

  # calculate IoU
  per_class_iou = np.divide(np.diag(conf), conf.sum(
      0) + conf.sum(1) - np.diag(conf))
  mean_iou = np.nanmean(per_class_iou)

  # calculate precision and recall
  per_class_prec = np.divide(np.diag(conf), conf.sum(0))
  per_class_rec = np.divide(np.diag(conf), conf.sum(1))

  # calculate kappa
  observed_acc = mean_acc
  expected_acc = np.divide(np.dot(conf.sum(0), conf.sum(1)), conf.sum()**2)
  kappa = np.divide(observed_acc - expected_acc, 1 - expected_acc)

  return mean_acc, mean_iou, per_class_iou, per_class_prec, per_class_rec, kappa
