#!/bin/bash

set -e
set -x
SRCDIR=`dirname $0`
CODEDIR=`dirname $SRCDIR`

# cd $CODEDIR

WORKDIR=`mktemp -d $SRCDIR/mt-dnn-tests-XXX`

mkdir -p $WORKDIR/mt_dnn_models
mkdir -p $WORKDIR/checkpoints

#download pre-trained BERT model
wget https://mrc.blob.core.windows.net/mt-dnn-model/bert_model_base_v2.pt -O $WORKDIR/mt_dnn_models/bert_model_base_uncased.pt

# download vocab file for BERT
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip -O "uncased_bert_base.zip"
unzip uncased_bert_base.zip
mv uncased_L-12_H-768_A-12/vocab.txt "$WORKDIR/mt_dnn_models/"
rm -rf uncased_bert_base.zip
rm -rf uncased_L-12_H-768_A-12


function delete {
    rm -rf $1
}


# tests begin here
i=0

for hparams in "" ; do

    # train
    python $CODEDIR/train.py --data_dir $CODEDIR/sample_data/output --init_checkpoint $WORKDIR/mt_dnn_models/bert_model_base_uncased.pt --batch_size 20 --batch_size_eval 20 --output_dir $WORKDIR/checkpoints/mt_dnn_results/ --log_file $WORKDIR/checkpoints/mt_dnn_results/log.log --answer_opt 0 --optimizer adamax --train_datasets mnli --test_datasets mnli_matched --grad_clipping 0 --global_grad_clipping 1 --learning_rate 5e-5 --log_per_updates 1 --epochs 2 --grad_accumulation_step 2

    # check if result files exist
    if [ ! -f $WORKDIR/checkpoints/mt_dnn_results/model_0.pt ] && [ ! -f $WORKDIR/checkpoints/mt_dnn_results/model_1.pt ]; then
        echo "Checkpoint files not found!"
        exit 1
    fi

    # load model and resume training
    python ./train.py --data_dir $CODEDIR/sample_data/output --init_checkpoint $WORKDIR/mt_dnn_models/bert_model_base_uncased.pt --batch_size 20 --batch_size_eval 20 --output_dir $WORKDIR/checkpoints/mt_dnn_results/ --log_file $WORKDIR/checkpoints/mt_dnn_results/log.log --answer_opt 0 --optimizer adamax --train_datasets mnli --test_datasets mnli_matched --grad_clipping 0 --global_grad_clipping 1 --learning_rate 5e-5 --log_per_updates 1 --epochs 2 --grad_accumulation_step 2 --resume --model_ckpt $WORKDIR/checkpoints/mt_dnn_results/model_1.pt


    i=$((i+1))
done

trap "delete $WORKDIR" TERM

