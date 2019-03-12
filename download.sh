#!/usr/bin/env bash
############################################################## 
# This script is used to download resources for MT-DNN experiments
############################################################## 

DATA_DIR=$(pwd)/data
echo "Create a folder $DATA_DIR"
mkdir ${DATA_DIR}

BERT_DIR=$(pwd)/mt_dnn_models
echo "Create a folder BERT_DIR"
mkdir ${BERT_DIR}

## DOWNLOAD GLUE DATA
## Please refer glue-baseline install requirments or other issues.
git clone https://github.com/jsalt18-sentence-repl/jiant.git
cd jiant
python scripts/download_glue_data.py --data_dir $DATA_DIR --tasks all

cd ..
rm -rf jiant
#########################

## DOWNLOAD SciTail 
cd $DATA_DIR
wget http://data.allenai.org.s3.amazonaws.com/downloads/SciTailV1.1.zip
unzip SciTailV1.1.zip
mv SciTailV1.1 SciTail
# remove zip files
rm *.zip

cd ${BERT_DIR}
## DOWNLOAD BERT
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -O "uncased_bert_base.zip"
unzip uncased_bert_base.zip
mv uncased_L-12_H-768_A-12/vocab.txt "${BERT_DIR}/"
rm *.zip
rm -rf uncased_L-12_H-768_A-12

## Download bert models
wget https://mrc.blob.core.windows.net/mt-dnn-model/bert_model_base_v2.pt -O "${BERT_DIR}/bert_model_base.pt"
wget https://mrc.blob.core.windows.net/mt-dnn-model/bert_model_large_v2.pt -O "${BERT_DIR}/bert_model_large.pt"
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_base.pt -O "${BERT_DIR}/mt_dnn_base.pt"
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_large.pt -O "${BERT_DIR}/mt_dnn_large.pt"

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
