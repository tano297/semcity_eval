#!/usr/bin/python3

import numpy as np
import argparse
import yaml
import os
import sys
from PIL import Image  # for tif image
import metrics
import builtins
import time

# redefine print so we can adjust verbosity
verbose = True


def set_verbosity(verbose_mode=True):
  """ Define verbosity
  """
  global verbose
  verbose = verbose_mode


def print(*args, **kwargs):
  """ Redefinition for verbose print
  """
  if verbose:
    builtins.print(*args, **kwargs)


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./evaluate_instance_segmentation.py")
  parser.add_argument(
      '--submission_dir', '-s',
      type=str,
      required=True,
      help='Directory where submitted predictions are.',
  )
  parser.add_argument(
      '--label_dir', '-l',
      type=str,
      required=True,
      help='Directory where the labels are.',
  )
  parser.add_argument(
      '--config', '-c',
      type=str,
      required=False,
      default="./classes.yaml",
      help='File where the config is for the classes.',
  )
  parser.add_argument(
      '--verbose', '-v',
      dest='verbose',
      default=False,
      action='store_true',
      help='Verbose mode. Print metrics on screen. Defaults to %(default)s',
  )
  FLAGS, unparsed = parser.parse_known_args()

  # set verbosity
  set_verbosity(FLAGS.verbose)

  # print summary of what we will do
  print("Parameters used:".center(80, "*"))
  print("Path to submission files: ", FLAGS.submission_dir)
  print("Path to label files", FLAGS.label_dir)
  print("Path to config file", FLAGS.config)
  print("Verbose?: ", FLAGS.verbose)
  print("End parameters used:".center(80, "*"))

  # try to open config yaml
  try:
    print("Opening config file: ", FLAGS.config)
    file = open(FLAGS.config, 'r')
    CFG = yaml.load(file)
    print("Success!")
  except Exception as e:
    print("Error!")
    print("  ↳", e)
    sys.exit(-1)

  # get number of classes from index dictionary
  num_classes = len(CFG["idx_to_name"])

  # file name stuff (files should be label_prefix_ + XX + .label_format)
  label_prefix = CFG["instance_label_prefix"]
  label_format = CFG["label_format"]

  # Check if submission and label directories exist, and get the labels
  if os.path.isdir(FLAGS.submission_dir):
    print("Submission dir ", FLAGS.submission_dir, " exists!")
    # get all files that have the prefix
    submissions = [f for f in os.listdir(FLAGS.submission_dir)
                   if os.path.isfile(os.path.join(FLAGS.submission_dir, f)) and
                   label_prefix in f and
                   label_format in f]
    # print(submissions)
  else:
    print("Submission dir ", FLAGS.submission_dir, " does not exist!")
    sys.exit(-1)

  if os.path.isdir(FLAGS.label_dir):
    print("Labels dir ", FLAGS.label_dir, " exists!")
    # get all files that have the prefix
    labels = [f for f in os.listdir(FLAGS.label_dir)
              if os.path.isfile(os.path.join(FLAGS.label_dir, f)) and
              label_prefix in f and
              label_format in f]
    # print(labels)
  else:
    print("Labels dir ", FLAGS.label_dir, " does not exist!")
    sys.exit(-1)

  # sort files
  submissions.sort()
  labels.sort()

  # track that each submission has a label and vice-versa, otherwise complain
  for sub in submissions:
    if sub not in labels:
      # warn that submission has no label
      print("Submission ", sub, " has no corresponding label")
      sys.exit(-1)
    else:
      print("Found Label for ", sub)

  for lbl in labels:
    if lbl not in submissions:
      # warn that label has no submission
      print("Label ", lbl, " has no corresponding submission")
      sys.exit(-1)
    else:
      print("Found Submission for ", lbl)

  # since lists are sorted, and we have 1 to 1 mapping for all files, and
  # knowing that 2 files don't have the same name, we can assume pass them
  # directly to the evaluator after opening them, and converting them to
  # instance polygons (it is expensive and we only want to do it once)
  sub_instances = []
  lbl_instances = []

  for s in submissions:
    # open submission, get instances, and put in submission structure
    sub = os.path.join(FLAGS.submission_dir, s)
    print("Extracting polygons from submission ", sub)
    sub_np = np.array(Image.open(sub)).astype(np.int32)
    sub_bboxes, sub_vals = metrics.get_instances(sub_np)
    sub_bboxes = metrics.inst_to_inst_struct(
        sub_bboxes, sub_vals, instance_type="prediction")

    # open label, get instances, and put in label structure
    lbl = os.path.join(FLAGS.label_dir, s)
    print("Extracting polygons from label ", lbl)
    lbl_np = np.array(Image.open(lbl)).astype(np.int32)
    lbl_bboxes, lbl_vals = metrics.get_instances(lbl_np)
    lbl_bboxes, lbl_strtree = metrics.inst_to_inst_struct(
        lbl_bboxes, lbl_vals, instance_type="label")

    # match the instances and fill in the iou metric for each connection
    time_start = time.time()
    for inst in sub_bboxes:
      metrics.match_instances(inst, lbl_bboxes, lbl_strtree, sub_np, lbl_np)
    elapsed = time.time() - time_start
    print("time to intersect ", elapsed)

    # append matched instances to each list
    sub_instances.append(sub_bboxes)
    lbl_instances.append(lbl_bboxes)

  # calculate all the metrics
  iou_range = np.arange(0.5, 0.96, 0.05)
  AP_50_95 = 0
  AP_50 = 0
  AP_75 = 0
  for iou in iou_range:
    AP, REC, TP, FP, FN = metrics.calculate_AP_for_iou_multi_image(
        sub_instances, lbl_instances, iou)
    AP_50_95 += AP
    print("===========")
    print("IoU: ", iou)
    print("AP: ", AP)
    print("Recall: ", REC)
    print("TP: ", TP, ", FP: ", FP, ", FN: ", FN)
    print("===========")

    if iou > 0.4 and iou < 0.6:
      AP_50 = AP
    elif iou > 0.74 and iou < 0.76:
      AP_75 = AP

  # average all the different IoU Average Precisions
  AP_50_95 /= iou_range.size

  # print if in verbose mode
  print("AP 50-95: ", AP_50_95)
  print("AP 50: ", AP_50)
  print("AP 75: ", AP_75)

  # in summary:
  # AP_50_95 is a scalar containing the mean Average Precision in IoU range [0.5:0.05:0.95]
  # AP_50 is a scalar containing the Average Precision for 0.5 IoU
  # AP_75 is a scalar containing the Average Precision for 0.75 IoU (harder)
