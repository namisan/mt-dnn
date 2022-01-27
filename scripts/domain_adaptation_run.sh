#!/bin/bash
if [[ $# -ne 8 ]]; then
  echo "train.sh <prefix> <bert_path> <train_datasets> <test_datasets> <data_dir> <model_dir> <batch_size> <gpu>"
  exit 1
fi
prefix=$1
BERT_PATH=$2
train_datasets=$3
test_datasets=$4
DATA_DIR=$5
MODEL_ROOT=$6
BATCH_SIZE=$7
gpu=$8
echo "export CUDA_VISIBLE_DEVICES=${gpu}"
export CUDA_VISIBLE_DEVICES=${gpu}
tstr=$(date +"%FT%H%M")

answer_opt=0
optim="adamax"
grad_clipping=0
global_grad_clipping=1

model_dir="checkpoints/${prefix}_${optim}_answer_opt${answer_opt}_gc${grad_clipping}_ggc${global_grad_clipping}_${tstr}"
log_file="${model_dir}/log.log"
python ../train.py --data_dir ${DATA_DIR} --init_checkpoint ${BERT_PATH} --batch_size ${BATCH_SIZE} --output_dir ${model_dir} --log_file ${log_file} --answer_opt ${answer_opt} --optimizer ${optim} --train_datasets ${train_datasets} --test_datasets ${test_datasets} --grad_clipping ${grad_clipping} --global_grad_clipping ${global_grad_clipping} --multi_gpu_on
