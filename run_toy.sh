#!/usr/bin/env bash

############################### 
# Training a mt-dnn model
# Note that this is a toy setting and please refer our paper for detailed hyper-parameters.
############################### 

# cook GLUE data
bash experiments/glue/prepro.sh

# FT on rte
export CUDA_VISIBLE_DEVICES=0,; bash experiments/glue/run_glue_finetuning.sh data/canonical_data/ bert base rte 16 1