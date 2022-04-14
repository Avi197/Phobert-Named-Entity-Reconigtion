import json
from pathlib import Path

import jsonlines


def txt_to_csv(file_path):
    out_path = file_path.replace('.txt', '.csv')
    with open(out_path, 'w', encoding='utf-8') as out_file:
        out_file.write('tokens,ner_tags\n')
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '' or line == '\n':
                    out_file.write('\n')
                else:
                    splits = line.split(' ')
                    tokens = splits[0]
                    ner_tags = splits[1].rstrip()
                    out_file.write(f'"{tokens}",{ner_tags}\n')


def txt_to_json(file_path):
    out_path = file_path.replace('.txt', '.json')
    # with open(out_path, 'w', encoding='utf-8') as out_file:
    with jsonlines.open(out_path, mode='w') as out_file:
        tokens = []
        ner_tags = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line == '' or line == '\n':

                    out_file.write({"tokens": tokens,
                                    "ner_tags": ner_tags})
                    tokens = []
                    ner_tags = []
                else:
                    splits = line.split(' ')
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            out_file.write({"tokens": tokens,
                            "ner_tags": ner_tags})


if __name__ == '__main__':
    l_train_path = '/home/phamson/data/tagging_stock_data/train.txt'
    l_dev_path = '/home/phamson/data/tagging_stock_data/dev.txt'
    l_test_path = '/home/phamson/data/tagging_stock_data/test.txt'

    # Path(l_train_path).unlink(missing_ok=True)
    # Path(l_dev_path).unlink(missing_ok=True)
    # Path(l_test_path).unlink(missing_ok=True)

    # txt_to_csv(l_train_path)
    # txt_to_csv(l_dev_path)
    # txt_to_csv(l_test_path)

    txt_to_json(l_train_path)
    txt_to_json(l_dev_path)
    txt_to_json(l_test_path)
