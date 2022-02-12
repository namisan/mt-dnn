#!/bin/bash

set -e
set -x
SRCDIR=`dirname $0`
CODEDIR=`dirname $SRCDIR`

WORKDIR=`mktemp -d $SRCDIR/mt-dnn-tests-XXX`

mkdir -p $WORKDIR/mt_dnn_models
mkdir -p $WORKDIR/checkpoints


function delete {
    rm -rf $1
}


# tests begin here
i=0

for hparams in "" ; do

    # train
    python $CODEDIR/train.py --data_dir $CODEDIR/tests/sample_data/output --task_def $CODEDIR/tests/mnli_task_def.yml --init_checkpoint bert-base-uncased --transformer_cache $WORKDIR/mt_dnn_models/cache --batch_size 20 --batch_size_eval 20 --bert_dropout_p 0 --output_dir $WORKDIR/checkpoints/mt_dnn_results/ --log_file $WORKDIR/checkpoints/mt_dnn_results/log.log --optimizer adamax --train_datasets mnli --test_datasets mnli_matched --learning_rate 5e-5 --log_per_updates 1 --epochs 2 --grad_accumulation_step 2

    # check if result files exist
    if [ ! -f $WORKDIR/checkpoints/mt_dnn_results/model_0.pt ] && [ ! -f $WORKDIR/checkpoints/mt_dnn_results/model_1.pt ]; then
        echo "Checkpoint files not found!"
        exit 1
    fi

    # load model and resume training
    python $CODEDIR/train.py --data_dir $CODEDIR/tests/sample_data/output --task_def $CODEDIR/tests/mnli_task_def.yml --init_checkpoint bert-base-uncased --transformer_cache $WORKDIR/mt_dnn_models/cache --batch_size 20 --batch_size_eval 20 --bert_dropout_p 0 --output_dir $WORKDIR/checkpoints/mt_dnn_results/ --log_file $WORKDIR/checkpoints/mt_dnn_results/log.log --optimizer adamax --train_datasets mnli --test_datasets mnli_matched --learning_rate 5e-5 --log_per_updates 1 --epochs 2 --grad_accumulation_step 2 --resume --model_ckpt $WORKDIR/checkpoints/mt_dnn_results/model_1.pt


    i=$((i+1))
done

trap "delete $WORKDIR" TERM

