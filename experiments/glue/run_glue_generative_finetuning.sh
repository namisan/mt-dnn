#!/usr/bin/env bash

############################### 
# Training script for generative GLUE.
# It supports single and multi-task training
# By Xiaodong 
############################### 

set -e


if [[ $# -lt 6 ]]; then
  echo "It requires 6 args to run the script and the current # of bash args: $#"
  echo "run_glue_finetune.sh <data_dir> <model_type> <model_size> <task> <batch-size> <num_gpus>"
  exit 1
fi

data_dir=$1
echo "Data dir: ${data_dir}"
model_type=$2
echo "Model type: ${model_type}"
model_size=$3
echo "Model size: ${model_size}"
# training set
task=$4
echo $task
batch_size=${5:-"16"}
num_gpus=${6:-"1"}
echo "GPU counts: ${num_gpus}"

export ROOT_DIR="glue_app_gen"
export EPOCH=3
export LR="2e-4"
export OPTIM="adamax"
export TASK_DEF="experiments/glue/glue_task_gen_def.yml"
export BS=${batch_size}

echo ${TASK_DEF}

train_dataset=${task}
test_dataset=${task}


# train task
if [ ${task} == "mnli" ]; then
test_dataset="mnli_matched,mnli_mismatched"
elif [ ${task} == "mtdnn" ]; then
train_dataset="mnli,rte,qqp,qnli,mrpc,sst,cola,stsb"
test_dataset="mnli_matched,mnli_mismatched,rte"
else
test_dataset=${task}
fi


echo "Training data: ${train_dataset}_train.json"
echo "Dev data: ${test_dataset}_dev.json"


if [ ${model_type} == "t5g" ]; then
  MD="t5-${model_size}"
  DD="t5-${model_size}"
  ED=9
  TOK="t5-base"
elif [ ${model_type} == "mistral" ]; then
  MD="mistralai/Mistral-7B-v0.1"
  DD="mistralai/Mistral-7B-v0.1"
  TOK="mistralai/Mistral-7B-v0.1"
  ED=14
elif [ ${model_type} == "mixtral" ]; then
  MD="mistralai/Mixtral-8x7B-v0.1"
  DD="mistralai/Mistral-7B-v0.1"
  TOK="mistralai/Mixtral-8x7B-v0.1"
  ED=15
else
  echo "Currently only support t5 generative finetuning"
  echo "Unknown model ${model_type}"
  exit 1
fi


output_dir="${ROOT_DIR}/${task}/${DD}"
echo $output_dir
mkdir -p ${output_dir}

if [[ -f "${output_dir}/model*.pt" ]]; then
 rm "${output_dir}/model*.pt"
 rm "${output_dir}/config.json"
fi

echo "Training ${task} tokenized by ${DD} with ${MD}"

LOG_FILE="${output_dir}/mt-dnn-train.log"
#
if [ ${num_gpus} -ge 2 ]; then
 # multi gpu training
 # DDP config
 export MASTER_ADDR=localhost
 export MASTER_PORT="8787"
 export NNODES=1
 export NODE_RANK=0
 export GPUS_PER_NODE=${num_gpus}
 export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
 export DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
 python -m torch.distributed.launch $DISTRIBUTED_ARGS train.py --data_dir=${data_dir}/${DD} --task_def=${TASK_DEF}  --train_dataset=${train_dataset} --test_dataset=${test_dataset} --init_checkpoint=${MD} --batch_size=${BS} --learning_rate=${LR} --epochs=${EPOCH} --encoder_type=${ED} --optimizer=${OPTIM} --output_dir=${output_dir} --log_file=${LOG_FILE} --tokenizer ${TOK}
else
 python train.py --data_dir=${data_dir}/${DD} --task_def=${TASK_DEF} --train_dataset=${train_dataset} --test_dataset=${test_dataset} --init_checkpoint=${MD} --batch_size=${BS} --learning_rate=${LR} --epochs=${EPOCH} --encoder_type=${ED} --optimizer=${OPTIM} --output_dir=${output_dir} --log_file=${LOG_FILE} --tokenizer ${TOK}
fi