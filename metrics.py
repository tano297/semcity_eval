#!/usr/bin/python3

import numpy as np
import cv2
import sys
from shapely import geometry
from shapely import strtree

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
  histrange = np.array([[-0.5, num_classes - 0.5],
                        [-0.5, num_classes - 0.5]], dtype='float64')
  confusion, _, _ = np.histogram2d(flat_label,
                                   flat_mask,
                                   bins=num_classes,
                                   range=histrange)
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

def inst_iou(sub_bbox, sub_val, sub_np, lbl_bbox, lbl_val, lbl_np):
  """ Given 2 bboxes and the images, calculate the Intersection over Union
  """

  # create common ROI with the union of the bboxes
  ROI = sub_bbox.union(lbl_bbox).bounds
  min_x = int(ROI[0])
  min_y = int(ROI[1])
  max_x = int(ROI[2])
  max_y = int(ROI[3])

  # extract ROI from images with the bboxes
  sub_ROI = sub_np[min_y:max_y, min_x:max_x]
  lbl_ROI = lbl_np[min_y:max_y, min_x:max_x]

  # mask to get IoU
  sub_mask = np.zeros(sub_ROI.shape)
  sub_mask[sub_ROI == sub_val] = 1
  lbl_mask = np.zeros(lbl_ROI.shape)
  lbl_mask[lbl_ROI == lbl_val] = 1

  # at this point, if there is no intersection, the sub_mask will be zeros
  if np.amax(sub_mask) < 1:
    return 0

  # get intersection over union
  conf = calculate_confusion(sub_mask, lbl_mask, num_classes=2)
  iou = np.divide(np.diag(conf), conf.sum(0) + conf.sum(1) - np.diag(conf))[1]

  return iou


def get_instances(image):
  """ Given an image where each instance is a different int value, and
      background is zero, return a list of bounding boxes and mask values
      where each instance is a different entity
  """
  bboxes = []
  vals = []

  # sanity check
  assert(len(image.shape) == 2)

  # instances are given from 1 to N, and may contain skipping values, so we
  # need a histogram of the pixels in the image and only add the ones containing
  # pixels to the range.
  val_range = []
  hist_bins = np.amax(image) + 1
  hist = np.bincount(image.flatten(), minlength=hist_bins)

  # get the actual buildings from histogram
  for i in range(1, hist_bins):  # start in 1 to ignore background
    if hist[i] > 0:
      val_range.append(i)

  # for each patch of value val, convert to bbox and append to lists
  for val in val_range:
    # get the mask
    bin_mask = cv2.compare(image, val, cv2.CMP_EQ)

    # convert to bbox
    instance_points = cv2.findNonZero(bin_mask)
    x, y, w, h = cv2.boundingRect(instance_points)

    # bbox to polygon
    bbox = geometry.box(x, y, x + w, y + h)

    # append to lists
    bboxes.append(bbox)
    vals.append(val)

  return bboxes, vals


def inst_to_inst_struct(bboxes, values, instance_type=""):
  """
    Fill the standard structure for instances depending on each class of
    instance
  """
  if instance_type == "label":
    # each label instance is a polygon, and a flag if it has been detected or not
    # as a dictionary
    struct_instances = [{"bbox": bbox, "val": val, "marked": False}
                        for bbox, val in zip(bboxes, values)]
    strtree_instances = strtree.STRtree(bboxes)
    return struct_instances, strtree_instances
  elif instance_type == "prediction":
    # each mask instance is a polygon
    struct_instances = [{"bbox": bbox, "val": val, "iou": 0, "label_bbox_idx": -1}
                        for bbox, val in zip(bboxes, values)]
    return struct_instances
  else:
    print("Invalid instance type: ", str(instance_type))
    sys.exit(-1)


def match_instances(instance, label_instances, label_instances_strtree, sub_np, lbl_np):
  """ Given an instance bbox, and a list of label bboxes and marks, along with the
      original images as a numpy array, return:
      The IoU with an instance of ground truth if it corresponds.
      The index of that ground truth index if it corresponds.
      Each polygon gets matched to the biggest corresponding IoU in the GT.
  """

  # query all labels that intersect with the instance
  intersection = label_instances_strtree.query(instance["bbox"])

  # get all struct instances that are in intersection. This is suboptimal, but
  # until I find a way to piggy back the info into the strtree I can't deal
  # with it differently
  instance_intersection = []
  for inst in label_instances:
    if inst["bbox"] in intersection:
      instance_intersection.append(inst)

  # check each label instance with the instance
  for lbl_instance in instance_intersection:
    # if iou with this instance is bigger than the IoU contained in the
    # prediction, match them
    iou = inst_iou(instance["bbox"], instance["val"], sub_np,
                   lbl_instance["bbox"], lbl_instance["val"], lbl_np)
    if iou > instance["iou"]:
      instance["iou"] = iou
      instance["label_bbox_idx"] = label_instances.index(lbl_instance)
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

  # reset the "marked" field for the labels
  for lbl in label:
    lbl["marked"] = False

  # for each prediction in the mask, check the IoU with the labels
  for instance in mask:
    # if the intersection IoU overlaps more than the threshold
    if instance["iou"] > IoU_th:
      # and the instance hasn't been detected yet, mark as true positive, and
      # mark the GT instance as already detected.
      if label[instance["label_bbox_idx"]]["marked"] is False:
        TP += 1
        label[instance["label_bbox_idx"]]["marked"] = True
      else:
        FP += 1
    else:
      FP += 1

  # as a last step, mark all the missed gt instances as False Negatives
  for instance in label:
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
