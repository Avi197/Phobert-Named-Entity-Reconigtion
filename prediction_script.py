import re
import torch
import json
import os
import shutil
import time
import glob
import pandas as pd
from datetime import datetime

import requests
from underthesea import sent_tokenize, word_tokenize
from transformers import AutoModelForTokenClassification, AutoTokenizer

REAL_ESTATE_LABEL_LIST = [
    "B-CNAME",
    "B-HNAME",
    "I-CNAME",
    "I-HNAME",
    "MCK",
    "O"
]


def replace_special_character(data):
    data = data.replace("​", ' ')
    data = data.replace("’", "'")
    data = data.replace("–", "-")
    return data


def remove_url(data):
    regex = re.compile(r'(?:http|ftp)s?://[a-z\d./-]+', re.IGNORECASE)
    return re.sub(regex, "", data)


def preprocess(text):
    content = text
    content = replace_special_character(content)
    paragraphs = [p for p in content.split("\n") if p != '']

    tokenize_sentences = []
    for paragraph in paragraphs:
        sentences = sent_tokenize(paragraph)
        for sentence in sentences:
            sentence = remove_url(sentence)
            if sentence != '':
                tokenize_sentence = word_tokenize(sentence, format="text")
                tokenize_sentences.append(tokenize_sentence)
    return tokenize_sentences


def get_bi_label(bi_label, text, label):
    if bi_label:
        return {"text": text, "label": label}
    else:
        return {"text": text, "label": label[2:] if label != 'O' else 'O'}


def clean_predictions(tokens, predictions, found_entity, bi_label=False):
    export_sentence = []
    temp_word = ''
    temp_label = 'O'
    entity = []
    entity_label = ''
    for token, prediction in zip(tokens, predictions[0].numpy()):
        label = REAL_ESTATE_LABEL_LIST[prediction]
        if token != '<s>' and token != '</s>':
            if '@@' in token:
                temp_word += token[:-2]
                if temp_label == 'O':
                    temp_label = label
            else:
                if temp_word != '':
                    text = temp_word + token
                    item = get_bi_label(bi_label, text, temp_label)
                    export_sentence.append(item)
                    temp_word = ''
                    temp_label = 'O'
                else:
                    text = token
                    item = get_bi_label(bi_label, text, label)
                    export_sentence.append(item)
                if label != 'O':
                    entity.append(text)
                    entity_label = label[2:]
                else:
                    if len(entity) != 0:
                        item = {"text": " ".join(entity), "label": entity_label}
                        if item not in found_entity:
                            found_entity.append(item)
                        entity = []
                        entity_label = ''
    return export_sentence, found_entity


def tagging(tokenize_sentences, result_path):
    entities = []
    with open(result_path, 'w') as f:
        for sentence in tokenize_sentences:
            try:
                list_ids = tokenizer(sentence)['input_ids']

                if len(list_ids) >= 256:
                    list_ids = list_ids[0:255]
                input_ids = torch.tensor([list_ids])

                tokens = tokenizer.convert_ids_to_tokens(list_ids)
                outputs = phobert_ner(input_ids).logits
                predictions = torch.argmax(outputs, dim=2)
                export_sentence, entities = clean_predictions(tokens, predictions, entities)

                for word in export_sentence:
                    f.write(f"{word['text']} {word['label']}\n")

                f.write('\n')
            except Exception as e:
                print(e)
    # print(export_sentence)


if __name__ == '__main__':
    bert_path = ''
    phobert_path = ''

    phobert_ner = AutoModelForTokenClassification.from_pretrained(bert_path)
    tokenizer = AutoTokenizer.from_pretrained(phobert_path, use_fast=False)

    torch.set_num_threads(2)
    from pathlib import Path

    text_path = ''
    out_path = ''
    count = 1
    for text_file in glob.glob(text_path + '/*.txt'):
        with open(text_file, 'r') as f:
            preprocessed = preprocess(f.read())
            tagging(preprocessed, os.path.join(out_path, f'{Path(text_file).name}'))
            print(count)
            count += 1

