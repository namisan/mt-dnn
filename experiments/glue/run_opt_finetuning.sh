#!/usr/bin/env bash

############################### 
# Training script for the 13B OPT model.
# It supports single and multi-task training
# By Xiaodong 
############################### 

## single task training
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3;
deepspeed --num_gpus=4 train.py \
--data_dir data/canonical_data/facebook/opt-13b/ \
--train_dataset mnli --test_dataset mnli_matched,mnli_mismatched \
--init_checkpoint facebook/opt-13b --tokenizer facebook/opt-13b --encoder_type 11 \
--task_def experiments/glue/glue_task_def.yml \
--batch_size 16 --batch_size_eval 16 \
--epochs 3 --max_seq_len 128 --learning_rate 1e-4 \
--deepspeed --deepspeed_config ds.json \
--log_per_updates 5


## multi-task training
export CUDA_VISIBLE_DEVICES=0,1,2,3;
deepspeed --num_gpus=4 train.py \
--data_dir data/canonical_data/facebook/opt-13b/ \
--train_dataset mnli,qqp --test_dataset mnli_matched,mnli_mismatched,qqp \
--init_checkpoint facebook/opt-13b --tokenizer facebook/opt-13b --encoder_type 11 \
--task_def experiments/glue/glue_task_def.yml \
--batch_size 16 --batch_size_eval 16 \
--epochs 3 --max_seq_len 128 --learning_rate 1e-4 \
--deepspeed --deepspeed_config ds.json \
--log_per_updates 5