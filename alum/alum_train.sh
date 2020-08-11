#!/bin/bash
# script to train an alum-roberta model.
# by Xiaodong Liu: xiaodl at microsoft.com
if [[ $# -ne 4 ]]; then
  echo "alum_train.sh <data_dir> <LR> <alum_code_path> <RoBERTa_model_path>"
  exit 1
fi
DATA_DIR=$1
PEAK_LR=$2
ALUM=$3
MODEL_PATH=$4
TOTAL_UPDATES=100000
WARMUP_UPDATES=10000
TOKENS_PER_SAMPLE=512
MAX_POSITIONS=512
MAX_SENTENCES=8
UPDATE_FREQ=2
WORKER=2

fairseq-train --fp16 $DATA_DIR \
    --task adv_masked_lm --criterion adv_masked_lm \
    --arch advbert_base --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-sentences $MAX_SENTENCES --update-freq $UPDATE_FREQ --restore-file ${MODEL_PATH} \
    --max-update $TOTAL_UPDATES --log-format simple --log-interval 500 --skip-invalid-size-inputs-valid-test --user-dir ${ALUM} --adv_alpha 10

