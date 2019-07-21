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

## Download bert models
wget https://mrc.blob.core.windows.net/mt-dnn-model/bert_model_base_v2.pt -O "${BERT_DIR}/bert_model_base_uncased.pt"
wget https://mrc.blob.core.windows.net/mt-dnn-model/bert_model_large_v2.pt -O "${BERT_DIR}/bert_model_large_uncased.pt"
wget https://mrc.blob.core.windows.net/mt-dnn-model/bert_base_chinese.pt -O "${BERT_DIR}/bert_model_base_chinese.pt"

## Download MT-DNN models
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_base.pt -O "${BERT_DIR}/mt_dnn_base_uncased.pt"
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_large.pt -O "${BERT_DIR}/mt_dnn_large_uncased.pt"

## MT-DNN-KD
wget https://mrc.blob.core.windows.net/mt-dnn-model/mt_dnn_kd_large_cased.pt -O "${BERT_DIR}/mt_dnn_kd_large_cased.pt"

## Download XLNet model
wget https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip -O "xlnet_cased_large.zip"
unzip xlnet_cased_large.zip 
mv xlnet_cased_L-24_H-1024_A-16/spiece.model -O "${BERT_DIR}/xlnet_large_cased_spiece.model"
rm -rf *.zip xlnet_cased_L-24_H-1024_A-16
## download converted xlnet pytorch model
wget https://mrc.blob.core.windows.net/mt-dnn-model/xlnet_model_large_cased.pt -O "${BERT_DIR}/xlnet_model_large_cased.pt"

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
