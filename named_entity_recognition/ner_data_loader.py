import os
from datasets import load_dataset

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

data_files = {}
data_files["train"] = train_file
data_files["validation"] = validation_file
data_files["test"] = test_file

extension = train_file.split(".")[-1]
raw_datasets = load_dataset('json', data_files=data_files)
