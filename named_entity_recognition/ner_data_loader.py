import os
from datasets import ClassLabel, load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


data_folder = '/opt/github/Phobert-Named-Entity-Reconigtion/tagging_stock_data'

train_file = os.path.join(data_folder, 'train.json')
validation_file = os.path.join(data_folder, 'dev.json')
test_file = os.path.join(data_folder, 'test.json')

# create dataset object for ner data
data_files = {"train": train_file,
              "validation": validation_file,
              "test": test_file}

# just set extension to json or csv or text according to data
extension = train_file.split(".")[-1]

# raw_dataset is now a dataset object
raw_datasets = load_dataset(f'{extension}', data_files=data_files)

column_names = raw_datasets["train"].column_names
features = raw_datasets["train"].features
# if 'tokens' in column_names:
#     text_column_name = "tokens"
# if 'ner_tags' in column_names:
#     label_column_name = 'ner_tags'

# set these according to data
label_column_name = 'ner_tags'
text_column_name = "tokens"

if isinstance(features[label_column_name].feature, ClassLabel):
    label_list = features[label_column_name].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
else:
    label_list = get_label_list(raw_datasets["train"][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)

# load model and tokenizer
model = AutoModelForTokenClassification.from_pretrained("vinai/phobert-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

# set label for model config
# model.config.label2id = {l: i for i, l in enumerate(label_list)}
# model.config.id2label = {i: l for i, l in enumerate(label_list)}

# map b (beginning) label with i (inner) label
b_to_i_label = []
for idx, label in enumerate(label_list):
    if label.startswith("B-") and label.replace("B-", "I-") in label_list:
        b_to_i_label.append(label.replace("B-", "I-"))
    else:
        b_to_i_label.append(idx)
