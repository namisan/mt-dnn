#!/bin/bash

SOURCE=$(dirname "$(readlink -f "$0")")/../src
export PYTHONPATH=${SOURCE}
SCRIPT=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT")
cd $SCRIPT_DIR

data_root=../data
BERT_BASE_DIR=../bert/uncased_L-24_H-1024_A-16
MODEL=$BERT_BASE_DIR/pytorch_model.bin
[ -e $MODEL ] || wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin -qO $MODEL
python -m pip install -r requirements.txt

CONFIG=hnn_config_large.json
VOCAB=$BERT_BASE_DIR/vocab.txt
Task=WSCR
Data=$data_root/DPRD
Epoch=8
BS=16
LR=1e-5
WRP=100
SEQ=128
LR_SCH=warmup_linear
Adam_Beta1=0.9
Adam_Beta2=0.999
QHAdam_V1=1
QHAdam_V2=1
GN=1
GNN=true
SEED=10037
CLS_DP=0.15
LSR=0
INIT_SPEC=hnn_init_large.spec
WNLIData=$data_root/WNLI
WikiData=$data_root/Wiki-WSCR
WSCData=$data_root/WSC273
GAPData=$data_root/GAP
GCC=1
ALPHA=0,0,0
BETA=0,0,0.6
GAMA=1,1,10
SIM='bilinear'
LOSS='binary'
TASKS='sm,lm'
GROUP_TASKS=True
EN=False
ATTENTIVE_POOL=False
POOL=cap #mean, ftp
SWA=0.95
SWA_START=2
TAG=wscr_SWA_${SWA_START}_${POOL}_gt_${GROUP_TASKS}_${TASKS}_${SIM}_${LOSS}_w${WRP}_${LR_SCH}_t${GAMA}_dprd_${ALPHA}_${BETA}_${GAMA}
OUTPUT=/data/experiments/$(hostname)/$Task/L/${Epoch}_${BS}_${LR}_${Adam_Beta1}_${QHAdam_V1}_${TAG}

[ -e $OUTPUT/script ] || mkdir -p $OUTPUT/script
cp -f $SCRIPT $OUTPUT/script
cp -f $CONFIG $OUTPUT/script
cp -f $INIT_SPEC $OUTPUT/script
cp -f $INIT_SPEC $OUTPUT/model_init.spec
cp -f $CONFIG $OUTPUT/bert_config.json
cp -f $VOCAB $OUTPUT/vocab.txt
rsync -ruzC --exclude-from=$SOURCE/.gitignore $SOURCE/ $OUTPUT/src

python $SOURCE/apps/run_hnn.py \
  --task_name $Task \
	--do_train \
  --do_lower_case \
  --data_dir $Data \
	--init_model $MODEL \
	--vocab $OUTPUT/vocab.txt \
	--bert_config $OUTPUT/bert_config.json \
  --max_seq_length $SEQ \
  --train_batch_size $BS \
  --eval_batch_size $BS \
  --learning_rate $LR \
  --num_train_epochs $Epoch \
  --output_dir $OUTPUT \
	--loss_scale 256 \
	--max_grad_norm ${GN} \
	--global_grad_norm ${GNN} \
	--scale_steps 500 \
	--seed $SEED \
	--tag $TAG \
	--lr_schedule $LR_SCH \
	--warmup_proportion $WRP \
	--worker_num 0 \
	--alpha=$ALPHA \
	--beta="$BETA" \
	--gama=$GAMA \
	--adam_beta1 $Adam_Beta1 \
	--adam_beta2 $Adam_Beta2 \
	--qhadam_v1 $QHAdam_V1 \
	--qhadam_v2 $QHAdam_V2 \
	--init_spec $OUTPUT/model_init.spec \
  --wsc273_data $WSCData \
	--similarity $SIM \
	--loss_type $LOSS \
	--tasks $TASKS \
	--group_tasks $GROUP_TASKS \
  --wnli_data $WNLIData \
	--ensemble $EN \
	--swa_start  $SWA_START \
	--pooling $POOL \
	--cls_drop_out ${CLS_DP}
