import logging
import os
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from utils_ner import Split, TokenClassificationDataset, TokenClassificationTask
from datasets import ClassLabel, load_dataset, load_metric

# base_model = "vinai/phobert-base"
#
# # phobert = AutoModel.from_pretrained(base_model)
# tokenizer = AutoTokenizer.from_pretrained(base_model)

# create data object
# training file location
# train_file = ''
# validation_file = ''
# test_file = ''
# # cache_dir location
# cache_dir = ''
#

# load data phai chay qua bo tokenizer de lay duoc token id

# data_files = {'train': train_file,
#               'validation': validation_file,
#               'test': test_file}
# extension = train_file.split('.')[-1]
# raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=cache_dir)

def convert_data():
