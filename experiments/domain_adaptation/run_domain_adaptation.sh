#!/usr/bin/env bash

############################### 
# Training script for domain adaptation
# By Xiaodong 
############################### 

set -e


if [[ $# -lt 5 ]]; then
  echo "run_domain_adaptation.sh <data_dir> <init_checkpoint> <train> <test> <batch-size>"
  exit 1
fi

data_dir=$1
ICKPT=$2
TRAIN=$3
TEST=$4
batch_size=${5:-"16"}

export ROOT_DIR="domain_app"
export EPOCH=3
export LR="5e-5"
export OPTIM="adamax"
export TASK_DEF="experiments/domain_adaptation/domain_adaptation_def.yml"
export BS=${batch_size}
export ED="1"
echo ${TASK_DEF}
task=$(echo ${TRAIN} | sed -e 's/_train.json//' )
echo $task


output_dir="${ROOT_DIR}/${task}"
echo $output_dir
mkdir -p ${output_dir}

if [[ -f "${output_dir}/model*.pt" ]]; then
  rm "${output_dir}/model*.pt"
  rm "${output_dir}/config.json"
fi

LOG_FILE="${output_dir}/domain-adaptation-train.log"

python train.py --data_dir=${data_dir}/${DD} --task_def=${TASK_DEF} --train_dataset=${TRAIN} --test_dataset=${TEST} --init_checkpoint=${ICKPT} --batch_size=${BS} --learning_rate=${LR} --epochs=${EPOCH} --encoder_type=${ED} --optimizer=${OPTIM} --output_dir=${output_dir} --log_file=${LOG_FILE}