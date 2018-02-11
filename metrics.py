#!/usr/bin/python3

import numpy as np
import cv2

################################################################################
#####################  Semantic Segmentation Metrics  ##########################


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

################################################################################


################################################################################
###################### Instance Segmentation Metrics ###########################

def polygon_iou(poly1, poly2):
  """ Given 2 polygons, calculate the Intersection over Union
  """
  iou = 0

  return iou


def get_instances(image):
  """ Given an image where each instance is a different int value, and
      background is zero, return a list of polygons where each instance is a
      polygon
  """
  instances = []

  # sanity check
  assert(len(image.shape) == 2)

  # instances are given from 1 to N-instance, with 0 being background.
  # therefore the max element will contain the max number of instances in the
  # image, which is the same as the number of polygons
  n_poly = np.amax(image)

  # for each patch of value val, convert to polygon and append to
  # list of polygons
  for val in range(1, n_poly):
    # get the mask
    poly_mask = np.zeros(image.shape, dtype=np.int8)
    poly_mask[image == val] = 1

    # convert to poly using contours
    _, poly, _ = cv2.findContours(
        poly_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    instances.append(poly)

  return instances


def match_instances(instance, label_instances):
  """ Given an instance polygon, and a list of polygons and marks, return:
      The IoU with an instance of ground truth if it corresponds.
      The index of that ground truth index if it corresponds.
      Each polygon gets matched to the biggest corresponding IoU in the GT.
  """

  # check each label instance with the instance
  for lbl_instance in label_instances:
    # if iou with this instance is bigger than the IoU contained in the
    # prediction, match them
    iou = polygon_iou(instance["poly"], lbl_instance["poly"])
    if iou > instance["iou"]:
      instance["iou"] = iou
      instance["label_poly_idx"] = label_instances.index(lbl_instance)
  return


def calculate_AP_for_iou_single_image(mask, label, IoU_th):
  """ Given prediction and label of a SINGLE CLASS detection problem, returns:
      The Average Precision for a certain IoU - AP = TP/(TP+FP)
      The Recall for a certain IoU - Rec = TP/(TP+FN)
      The amount of True Positives for a certain IoU - TP
      The amount of False Positives for a certain IoU - FP
      The amount of False Negatives for a certain IoU - FN
  """
  AP = 0
  REC = 0
  TP = 0
  FP = 0
  FN = 0

  # get each instance from the label image
  # each instance is a polygon, and a flag if it has been detected or not as
  # a dictionary
  label_instances = get_instances(label)
  for poly in label_instances:
    poly = {"polygon": poly, "marked": False}

  # get each instance from the mask (prediction) image
  # each instance is a polygon
  mask_instances = get_instances(mask)
  for poly in mask_instances:
    poly = {"polygon": poly, "iou": 0, "label_poly_idx": -1}

  # for each prediction in the mask, check the IoU with the labels
  for instance in mask_instances:
    # match the instance to the gt instances if possible
    match_instances(instance, label_instances)

    # if the intersection IoU overlaps more than the threshold
    if instance["iou"] > IoU_th:
      # and the instance hasn't been detected yet, mark as true positive, and
      # mark the GT instance as already detected.
      if label_instances[instance["label_poly_idx"]]["marked"] is False:
        TP += 1
        label_instances[instance["label_poly_idx"]]["marked"] = True
      else:
        FP += 1
    else:
      FP += 1

  # as a last step, mark all the missed gt instances as False Negatives
  for instance in label_instances:
    if instance["marked"] is False:
      FN += 1

  # calculate Average Precision and Recall for this image
  AP = TP / (TP + FP)
  REC = TP / (TP + FN)

  return AP, REC, TP, FP, FN


def calculate_AP_for_iou_multi_image(masks, labels, IoU_th):
  """ Given lists of predictions and labels of a SINGLE CLASS detection problem,
      it returns:
      The Average Precision for a certain IoU - AP = TP/(TP+FP)
      The Recall for a certain IoU - Rec = TP/(TP+FN)
      The amount of True Positives for a certain IoU
      The amount of False Positives for a certain IoU
      The amount of False Negatives for a certain IoU
  """
  AP = 0
  REC = 0
  tot_TP = 0
  tot_FP = 0
  tot_FN = 0

  # for each image and label, calculate TP, FP, FN for a single IoU
  for mask, lbl in zip(masks, labels):
    _, _, TP, FP, FN = calculate_AP_for_iou_single_image(mask, lbl, IoU_th)
    tot_TP += TP
    tot_FP += FP
    tot_FN += FN

  # calculate Average Precision and Recall for this image
  AP = tot_TP / (tot_TP + tot_FP)
  REC = tot_TP / (tot_TP + tot_FN)

  return AP, REC, tot_TP, tot_FP, tot_FN

################################################################################
