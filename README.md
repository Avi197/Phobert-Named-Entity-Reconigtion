# Phobert Named Entity Reconigtion
Using [Phobert](https://github.com/VinAIResearch/PhoBERT#-using-phobert-with-transformers) model by [VinAI Research](https://github.com/VinAIResearch) for NER task on various datasets

### Phobert with ```transformers```
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

### Results
Using only phobert-base, NER task return __F1 = 94.7%__


### Tokenization
Data must be tokenized before fine-tune
Using VnCoreNLP's word segmenter to pre-process input raw texts

A word segmenter must be applied to produce word-segmented texts before feeding to PhoBERT.\
As PhoBERT employed the [RDRSegmenter](https://github.com/datquocnguyen/RDRsegmenter) from [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) to pre-process the pre-training data

#### Installation
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

#### Example usage
```
# See more details at: https://github.com/vncorenlp/VnCoreNLP

# Load rdrsegmenter from VnCoreNLP
from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("/Absolute-path-to/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

# Input 
text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."

# To perform word (and sentence) segmentation
sentences = rdrsegmenter.tokenize(text) 
for sentence in sentences:
	print(" ".join(sentence))
```

### Run fine-tuning for NER task
The config.json file contain the arguments for run_ner, you can change the parameters in the config file or just run from command line

json config
```
{
    "data_dir": "bert-ner/data",
    "model_name_or_path": "vinai/phobert-base",
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

#### VLSP2016 NER data
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
là |	V |	B-VP |	O |	O
cán_bộ |	N |	B-NP |	O |	O
Uỷ ban |	N |	B-NP |	B-ORG |	O
nhân_dân |	N |	I-NP |	I-ORG |	O
Thành_phố |	N |	I-NP |	I-ORG |	B-LOC
Hà_Nội |	Np |	I-NP |	I-ORG |	I-LOC
. |	. |	O |	O |	O


This script only use word and NE column so the training data look like this

word | NE
--- | ---
Anh | O 
Thanh | B-PER
là | O
cán_bộ | O
Uỷ ban | B-ORG
nhân_dân | I-ORG
Thành_phố | I-ORG
Hà_Nội | I-ORG
. | O


preprocess the VLSP2016 dataset with ```preprocessing.py``` script to get the training data format
### Custom NER dataset

if you want to use with custom dataset, the data should be formatted as
```
word(space)tag
```

e.g.:
```
tắc đường ở Hà_Nội

tắc 0
đường 0
ở 0
Hà_Nội B_LOC
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

If you want to get access to [VLSP2016 dataset](https://vlsp.org.vn/resources-vlsp2016), sign the user agreement and mail to VLSP association

