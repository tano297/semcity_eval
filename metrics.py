#!/usr/bin/python3

import numpy as np
import cv2
import sys
import time
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
  # time_start = time.time()

  # get intersection over union
  try:
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
  except Exception:
    # invalid geometry
    return 0

  # valid geometry
  if union > 0:
    iou = intersection / union
  else:
    iou = 0

  # elapsed = time.time() - time_start
  # print("elapsed iou", elapsed)

  return iou


def get_instances(image):
  """ Given an image where each instance is a different int value, and
      background is zero, return a list of polygons where each instance is a
      polygon
  """
  instances = []

  # sanity check
  assert(len(image.shape) == 2)

  # instances are given from 1 to N, and may contain skipping values, so we
  # need a histogram of the pixels in the image and only add the ones containing
  # pixels to the range.
  poly_range = []
  hist_bins = np.amax(image) + 1
  hist = np.bincount(image.flatten(), minlength=hist_bins)

  # get the actual buildings from histogram
  for i in range(1, hist_bins):  # start in 1 to ignore background
    if hist[i] > 0:
      poly_range.append(i)

  # for each patch of value val, convert to polygon and append to
  # list of polygons
  for val in poly_range:
    # time_start = time.time()
    # get the mask
    poly_mask = cv2.compare(image, val, cv2.CMP_EQ)

    # convert to poly using contours
    img, poly, _ = cv2.findContours(
        poly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ignore all poligons with less than 3 coordinates.
    try:
      shapely_poly = np.reshape(poly[0], [poly[0].shape[0], poly[0].shape[2]])
      shapely_poly = geometry.Polygon(shapely_poly)

      # if it is not valid, validate with buffer(0) method
      if shapely_poly.is_valid:
        instances.append(shapely_poly)
      else:
        shapely_poly = shapely_poly.buffer(0)
        if shapely_poly.is_valid:
          instances.append(shapely_poly)
    except Exception:
      print("Warning: Ignoring polygon: ",
            poly[0], " because it is not a valid geometry")

    # elapsed = time.time() - time_start
    # print("poly time: ", elapsed)

    # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    # print("contour type", type(poly))
    # print("contour: ", poly)
    # cv2.drawContours(img, poly, -1, (0, 255, 0), 50)

    # cv2.imshow("img", img)
    # cv2.waitKey(1)

  return instances


def poly_to_poly_struct(instances, instance_type=""):
  """
    Fill the standard structure for instances depending on each class of
    instance
  """
  if instance_type == "label":
    # each label instance is a polygon, and a flag if it has been detected or not
    # as a dictionary
    struct_instances = [{"polygon": poly, "marked": False}
                        for poly in instances]
    strtree_instances = strtree.STRtree(instances)
    return struct_instances, strtree_instances
  elif instance_type == "prediction":
    # each mask instance is a polygon
    struct_instances = [{"polygon": poly, "iou": 0, "label_poly_idx": -1}
                        for poly in instances]
    return struct_instances
  else:
    print("Invalid instance type: ", str(instance_type))
    sys.exit(-1)


def match_instances(instance, label_instances, label_instances_strtree):
  """ Given an instance polygon, and a list of polygons and marks, return:
      The IoU with an instance of ground truth if it corresponds.
      The index of that ground truth index if it corresponds.
      Each polygon gets matched to the biggest corresponding IoU in the GT.
  """

  # query all labels that intersect with the instance
  intersection = label_instances_strtree.query(instance["polygon"])

  # get all struct instances that are in intersection. This is suboptimal, but
  # until I find a way to piggy back the info into the strtree I can't deal
  # with it differently
  instance_intersection = []
  for inst in label_instances:
    if inst in intersection:
      instance_intersection.append(inst)

  # check each label instance with the instance
  for lbl_instance in instance_intersection:
    # if iou with this instance is bigger than the IoU contained in the
    # prediction, match them
    iou = polygon_iou(instance["polygon"], lbl_instance["polygon"])
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

  # reset the "marked" field for the labels
  for lbl in label:
    lbl["marked"] = False

  # for each prediction in the mask, check the IoU with the labels
  for instance in mask:
    # if the intersection IoU overlaps more than the threshold
    if instance["iou"] > IoU_th:
      # and the instance hasn't been detected yet, mark as true positive, and
      # mark the GT instance as already detected.
      if label[instance["label_poly_idx"]]["marked"] is False:
        TP += 1
        label[instance["label_poly_idx"]]["marked"] = True
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
