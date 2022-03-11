#!/usr/bin/env bash
############################################################## 
# This script is used to download resources for MT-DNN experiments
############################################################## 

BERT_DIR=$(pwd)/mt_dnn_models
if [ ! -d ${BERT_DIR}  ]; then
  echo "Create a folder BERT_DIR"
  mkdir ${BERT_DIR}
fi

## Download MT-DNN models
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_base.pt -O "${BERT_DIR}/mt_dnn_base_uncased.pt"
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_large.pt -O "${BERT_DIR}/mt_dnn_large_uncased.pt"

## MT-DNN-KD
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_kd_large_cased.pt -O "${BERT_DIR}/mt_dnn_kd_large_cased.pt"


if [ "$1" == "model_only" ]; then
  exit 1
fi

DATA_DIR=$(pwd)/data
if [ ! -d ${DATA_DIR}  ]; then
  echo "Create a folder $DATA_DIR"
  mkdir ${DATA_DIR}
fi

## DOWNLOAD GLUE DATA
## Please refer glue-baseline install requirments or other issues.
git clone https://github.com/nyu-mll/jiant-v1-legacy.git
cd jiant-v1-legacy
python scripts/download_glue_data.py --data_dir $DATA_DIR --tasks all

cd ..
rm -rf jiant-v1-legacy
#########################

## DOWNLOAD SciTail
cd $DATA_DIR
wget http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.1.zip
unzip SciTailV1.1.zip
mv SciTailV1.1 SciTail
# remove zip files
rm *.zip

## Download preprocessed SciTail/SNLI data for domain adaptation
cd $DATA_DIR
DOMAIN_ADP="domain_adaptation"
echo "Create a folder $DATA_DIR"
mkdir ${DOMAIN_ADP}

wget https://mrc.blob.core.windows.net/mt-dnn-model/data.zip 
unzip data.zip
mv data/* ${DOMAIN_ADP}
rm -rf data.zip
rm -rf data

## Download SQuAD & SQuAD v2.0 data
cd $DATA_DIR
mkdir "squad"
cd ..
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json -O squad/train.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json -O squad/dev.json

mkdir "squad_v2"
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O squad_v2/train.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad_v2/dev.json

# NER
mkdir "ner"
wget https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train -O "ner/train.txt"
wget https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa -O "ner/valid.txt"
wget https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb -O "ner/test.txt"

# SuperGLUE

wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip  -O superglue.zip 
unzip superglue.zip
cd ..
