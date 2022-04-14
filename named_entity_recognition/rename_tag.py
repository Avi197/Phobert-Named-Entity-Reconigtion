import csv
import glob
import random
import math
from pathlib import Path
from itertools import groupby


def split_data_folder(training_folder, percent=30):
    list_train_total = []
    list_test_total = []
    for folder in glob.glob(training_folder + '/**'):
        list_tsv = glob.glob(folder + '/*.tsv')
        list_test = random.sample(list_tsv, math.floor(len(list_tsv) * percent / 100))
        # print(len(list_tsv))
        # print(len(list_test))
        # print('------------')
        list_train = [x for x in list_tsv if x not in list_test]
        list_train_total.append(list_train)
        list_test_total.append(list_test)

    # list_train_total = [item for sublist in list_train_total for item in sublist]
    # list_test_total = [item for sublist in list_test_total for item in sublist]

    list_train_total = flatten_list(list_train_total)
    list_test_total = flatten_list(list_test_total)
    return list_train_total, list_test_total


def split_data_list(split_list, percent=50):
    list_dev = random.sample(split_list, math.floor(len(split_list) * percent / 100))
    list_test = [x for x in split_list if x not in list_dev]
    return list_dev, list_test


def flatten_list(label_col_mod):
    flatten = [item for sublist in label_col_mod for item in sublist]
    return flatten


def convert_bert_format(tsv_file_list, output_file):
    with open(output_file, 'a', newline='') as output_f:
        for file in tsv_file_list:
            with open(file, 'r') as fd:
                # get 2 seperate text, label col
                text_col = []
                label_col = []
                rd = csv.reader(fd, delimiter="\t", quotechar='"')
                for idx, row in enumerate(rd):
                    if row:
                        text_col.append(row[1])
                        label_col.append(row[-1])
                    else:
                        text_col.append('')
                        label_col.append('')

            # group chunk of label
            label_group = [list(y) for _, y in groupby(label_col)]

            # change label
            label_col_mod = []
            for group in label_group:
                if not all(ele == 'O' for ele in group):
                    if any('H' in ele for ele in group):
                        group = [w.replace('H-', 'I-H') for w in group]
                        group[0] = group[0].replace('I-H', 'B-H')

                    elif any('C' in ele for ele in group):
                        group = [w.replace('C-', 'I-C') for w in group]
                        group[0] = group[0].replace('I-C', 'B-C')

                label_col_mod.append(group)

            # flatten grouped list
            flat_label = flatten_list(label_col_mod)

            # merge 2 list back
            res_list = [[t, l] for t, l in zip(text_col, flat_label)]
            for word in res_list:
                if all(x == '' for x in word):
                    output_f.write('\n')
                else:
                    output_f.write(f'{word[0]} {word[1]}\n')


if __name__ == '__main__':
    # files = glob.glob(training_data + '/**/*.tsv', recursive=True)
    # train_file = '/home/phamson/data/tagging_stock_data/train.txt'
    # test_file = '/home/phamson/data/tagging_stock_data/test.txt'

    training_data = '/home/phamson/gitlab/vnd-ai/vnd-news-nlp/data/trainingData'

    l_train, l_test_temp = split_data_folder(training_data)
    l_dev, l_test = split_data_list(l_test_temp)

    l_train_path = '/home/phamson/data/tagging_stock_data/train.txt'
    l_dev_path = '/home/phamson/data/tagging_stock_data/dev.txt'
    l_test_path = '/home/phamson/data/tagging_stock_data/test.txt'

    Path(l_train_path).unlink(missing_ok=True)
    Path(l_dev_path).unlink(missing_ok=True)
    Path(l_test_path).unlink(missing_ok=True)

    # l_train, l_dev, l_test
    convert_bert_format(l_train, l_train_path)
    convert_bert_format(l_dev, l_dev_path)
    convert_bert_format(l_test, l_test_path)
