# Phobert Named Entity Reconigtion
Using [Phobert](https://github.com/VinAIResearch/PhoBERT#-using-phobert-with-transformers) model by [VinAI Research](https://github.com/VinAIResearch) for NER task on various datasets

Tested on VLSP2016 test set using PhoBERT-base yield <b>94.7 F1 score</b>

Check the [notebook](https://github.com/Avi197/Phobert-Named-Entity-Reconigtion/blob/main/prediction.ipynb) for more detail

Implemented in [undertheseanlp](https://github.com/undertheseanlp) NER task

## Phobert with ```transformers```
#### Installation
* Python 3.6+, Pytorch 1.1.0+ (or TensorFlow 2.0+)
* Install ```transformers```:
	* ```git clone https://github.com/huggingface/transformers.git```
	* ```cd transformers```
	* ```pip3 install --upgrade```

#### Pre-trained models
Model | #params | Arch. | Pre-training data
------------ | ------------- | ------------- | -------------
```vinai/phobert-base``` | 135M | base | 20GB of texts
```vinai/phobert-large``` | 370M | large | 20GB of texts

You can fine-tune with either large model or base model , with base model train faster but large model give better result. \
For general use, just fine-tune on base model as large model only give a slightly better result

## Tokenization
Data must be tokenized before fine-tune
Using VnCoreNLP's word segmenter to pre-process input raw texts

A word segmenter must be applied to produce word-segmented texts before feeding to PhoBERT.\
As PhoBERT employed the [RDRSegmenter](https://github.com/datquocnguyen/RDRsegmenter) from [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) to pre-process the pre-training data

#### VncoreNLP installation
```
# Install the vncorenlp python wrapper
pip3 install vncorenlp

# Download VnCoreNLP-1.1.1.jar & its word segmentation component (i.e. RDRSegmenter) 
mkdir -p vncorenlp/models/wordsegmenter
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
mv VnCoreNLP-1.1.1.jar vncorenlp/ 
mv vi-vocab vncorenlp/models/wordsegmenter/
mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/
```

#### Tokenization xample usage
```
# See more details at: https://github.com/vncorenlp/VnCoreNLP

# Load rdrsegmenter from VnCoreNLP
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("/Absolute-path-to/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

# Input 
text = "??ng Nguy???n Kh???c Ch??c  ??ang l??m vi???c t???i ?????i h???c Qu???c gia H?? N???i. B?? Lan, v??? ??ng Ch??c, c??ng l??m vi???c t???i ????y."

# To perform word (and sentence) segmentation
sentences = rdrsegmenter.tokenize(text) 
for sentence in sentences:
	print(" ".join(sentence))
```

## Run fine-tuning for NER task
The config.json file contain the arguments for run_ner, you can change the parameters in the config file or just run from command line

json config
```
{
    "data_dir": "bert-ner/data",
    "model_name_or_path": "vinai/phobert-base",
    "labels": "label.txt path",
    "output_dir": "phobert-ner",
    "max_seq_length": 128,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 32,
    "save_steps": 750,
    "seed": 1,
    "do_train": true,
    "do_eval": true,
    "do_predict": true
}
python3 run_ner.py config.json
```

command line
```
python run_ner.py \
--data_dir /bert-ner/data \
--model_name_or_path vinai/phobert-base \
--labels label_path \
--output_dir phobert-ner \
--max_seq_length 128 \
--num_train_epochs 3 \
--per_device_train_batch_size 32 \
--save_steps 750 \
--seed 1 \
--do_train true \
--do_eval true \
--do_predict true \
```

### VLSP2016 NER data
Data have been preprocessing with word segmentation and POS tagging. The data consist of five columns, in which two columns are separated by a single space. Each word has been put on a separate line and there is an empty line after each sentence.

1. The first column is the word
2. The second column is its POS tag
3. The third column is its chunking tag
4. The fourth column is its NE label
5. The fifth column is its nested NE label

word | POS | chunking | NE | nested NE
---| --- | --- | --- | ---
Anh |	N |	B-NP |	O |	O
Thanh |	Np |	I-NP |	B-PER |	O
l?? |	V |	B-VP |	O |	O
c??n_b??? |	N |	B-NP |	O |	O
U??? ban |	N |	B-NP |	B-ORG |	O
nh??n_d??n |	N |	I-NP |	I-ORG |	O
Th??nh_ph??? |	N |	I-NP |	I-ORG |	B-LOC
H??_N???i |	Np |	I-NP |	I-ORG |	I-LOC
. |	. |	O |	O |	O


This script only use word and NE column so the training data look like this

word | NE
--- | ---
Anh | O 
Thanh | B-PER
l?? | O
c??n_b??? | O
U??? ban | B-ORG
nh??n_d??n | I-ORG
Th??nh_ph??? | I-ORG
H??_N???i | I-ORG
. | O


preprocess the VLSP2016 dataset with ```preprocessing.py``` script to get the training data format
### Custom NER dataset

if you want to use with custom dataset, the data should be formatted as
```
word(space)tag
```

e.g.:
```
t???c ???????ng ??? H??_N???i

t???c 0
???????ng 0
??? 0
H??_N???i B_LOC
```

Get the unique labels from data
```
cat train.txt dev.txt test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > labels.txt
```

the tagging follow conll2003 dataset as
```
"O"        # Outside of a named entity
"B-MISC"   # Beginning of a miscellaneous entity right after another miscellaneous entity
"I-MISC"   # Miscellaneous entity
"B-PER"    # Beginning of a person's name right after another person's name
"I-PER"    # Person's name
"B-ORG"    # Beginning of an organisation right after another organisation
"I-ORG"    # Organisation
"B-LOC"    # Beginning of a location right after another location
"I-LOC"    # Location
```

clean up multiple blank lines
```
awk '!NF {if (++n <= 1) print; next}; {n=0;print}' train.txt > train_clean.txt
awk '!NF {if (++n <= 1) print; next}; {n=0;print}' dev.txt > dev_clean.txt
awk '!NF {if (++n <= 1) print; next}; {n=0;print}' test.txt > test_clean.txt
```


# Acknowledgements
Pretrained model [Phobert](https://github.com/VinAIResearch/PhoBERT#-using-phobert-with-transformers) by [VinAI Research](https://github.com/VinAIResearch)
NER dataset from [VLSP2016 dataset](https://vlsp.org.vn/resources-vlsp2016)

If you want to get access to [VLSP2016 dataset](https://vlsp.org.vn/resources-vlsp2016), sign the user agreement and mail to VLSP association

### If this project help you, please give me a star :star2: :star2: :star2:
