#!/usr/bin/python3

import numpy as np
import argparse
import yaml
import os
import sys
from PIL import Image  # for tif image
import metrics
import builtins

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
  parser = argparse.ArgumentParser("./evaluate_semantic_segmentation.py")
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
  label_prefix = CFG["semantic_label_prefix"]
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
  # knowing that 2 files don't have the same name, we can assume calculate
  # confusion matrix for each file, and later sum them and get the matrix
  confusion_matrixes = []
  confusion_matrix = np.zeros((num_classes, num_classes))

  for s in submissions:
    print("Calculating metrics for image ", s)
    # open submission
    sub_tif = Image.open(os.path.join(FLAGS.submission_dir, s))
    # open label
    lbl_tif = Image.open(os.path.join(FLAGS.label_dir, s))
    # calculate and append confusion matrix
    conf = metrics.calculate_confusion(
        mask=sub_tif, lbl=lbl_tif, num_classes=num_classes)
    # print(conf)
    confusion_matrixes.append(conf)
    confusion_matrix += conf

  # calculate all the metrics from confusion matrix
  mean_acc, mean_iou, per_class_iou, per_class_prec, per_class_rec, kappa = metrics.pix_metrics_from_confusion(
      confusion_matrix)

  # print if in verbose mode
  print("Overall Performance Metrics: ")
  print('   Mean Accuracy: %0.04f' % mean_acc)
  print('   Mean IoU: %0.04f' % mean_iou)
  print('   Kappa: %0.04f' % kappa)
  print('Class-wise Performance Metrics')
  for idx in range(num_classes):
    class_str = CFG["idx_to_name"][idx]
    print('   Class %d: %s' % (idx, class_str))
    print('      IoU: %0.04f' % (per_class_iou[idx]))
    print('      Precision: %0.04f' % (per_class_prec[idx]))
    print('      Recall: %0.04f' % (per_class_rec[idx]))

  # in summary:
  # confusion_matrixes = 1 confusion image per patch [list of np.array matrix]
  # confusion_matrix = sum of all the confusion matrixes of all the patches [np.array matrix]
  # mean_acc = mean pixel-wise accuracy (correct pixels / all) [scalar]
  # mean_iou = mean intersection over union (IoU, or Jaccard Index) [scalar]
  # per_class_iou = pixel-wise intersection over union of each class []
  # per_class_prec = pixel-wise precision of each class
  # per_class_rec = pixel-wise
  # kappa = kappa coefficient for the classifier
