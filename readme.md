# Semcity Evaluation API

This repository includes the evaluation scripts for both Semcity challenges:
  - Multi-Class Semantic Segmentation
  - Instance Segmentation of Buildings

If the directory with the predictions works on the corresponding script, then its 
zipped version will also work on the evaluation server. In order to try the
scripts, we suggest using either the training and validation labels to evaluate
the metrics, or use the test data submission directory against itself to check
if the evaluation server will run.

### Semantic Segmentation Task

To run the Semantic Segmentation evaluation (_-v_ for verbose):

```sh
  $ ./evaluate_semantic_segmentation.py  -s /path/to/prediction/dir -l /path/to/label/dir -v
```

### Instance Segmentation Task

To run the Instance Segmentation evaluation (_-v_ for verbose):

```sh
  $ ./evaluate_instance_segmentation.py  -s /path/to/prediction/dir -l /path/to/label/dir -v
```

### Requirements

In order to install the requirements needed for this script (which are listed
in the included _"requirements.txt"_ file):

```sh
  $ sudo pip3 install -r requirements.txt
```