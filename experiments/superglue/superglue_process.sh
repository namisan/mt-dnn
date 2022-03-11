#!/bin/bash
# Reuse of GLUE process script
# Copyright (c) Microsoft, Inc. and its affiliates.
#
# by Xiaodong Liu 
# xiaodl@microsoft.com
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

# This script is used to cook SuperGLUE data in FairSEQ format. 
# 
# offical data from SuperGLUE team is located: https://super.gluebenchmark.com/tasks
# ***Download***
# wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip
# unzip combined.zip

if [[ $# -ne 4 ]]; then
  echo "Run as following:"
  echo "process.sh <glud_data_folder> <task_name> <dict_dir> <output>"
  exit 1
fi

SUPERGLUE_DATA_FOLDER=$1
# e.g., BoolQ
TASKS=$2 

DICT=$3

OUTPUT=$4

mkdir -p $OUTPUT

if [ "$TASKS" = "ALL" ]
then
  TASKS="BoolQ MultiRC BC ReCoRD COPA WiC WSC"
  INPUT_COUNT=2
fi

INPUT_COUNT=2
for TASK in $TASKS
do
  echo "Preprocessing $TASK"

  TASK_DATA_FOLDER="$SUPERGLUE_DATA_FOLDER/$TASK"
  echo "Raw data as downloaded from glue website: $TASK_DATA_FOLDER"

  SPLITS="train val test"

  if [ "$TASK" = "MultiRC" ]
  then
  INPUT_COUNT=3
  fi

  if [ "$TASK" = "WiC" ]
  then
  INPUT_COUNT=3
  fi

  if [ "$TASK" = "ReCoRD" ]
  then
  INPUT_COUNT=3
  fi

  if [ "$TASK" = "COPA" ]
  then
  INPUT_COUNT=3
  fi

  # Strip out header and filter lines that don't have expected number of fields.
  rm -rf "$TASK_DATA_FOLDER/processed" ||:
  mkdir -p "$TASK_DATA_FOLDER/processed"
  for SPLIT in $SPLITS
  do
    # CoLA train and dev doesn't have header.
    cp "$TASK_DATA_FOLDER/$SPLIT.jsonl" "$TASK_DATA_FOLDER/processed/$SPLIT.jsonl";
  done

  # Split into input0, input1 and label
  python superglue_fairseq.py --data_dir $TASK_DATA_FOLDER/processed --task $TASK
  for SPLIT in $SPLITS
  do  
    echo ${SPLIT}
    echo $(seq 0 $((INPUT_COUNT-1)))
    # BPE encode.
    for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
    do
      MYLANG="input$INPUT_TYPE"
      echo "BPE encoding $SPLIT/$MYLANG"
      ## ***sentencepiece tokenizer for TNLR models***

      #cat $TASK_DATA_FOLDER/processed/$SPLIT.raw.$MYLANG | \
      #    python ../pretrain/multiprocessing_sp_encoder.py \
      #      --sentencepiece-model $DICT/sp.model \
      #      --vocab $DICT/dict.txt \
      #    > $TASK_DATA_FOLDER/processed/$SPLIT.$MYLANG

      ## bpe for RoBERTa
      python -m examples.roberta.multiprocessing_bpe_encoder \
      --encoder-json encoder.json \
      --vocab-bpe vocab.bpe \
      --inputs "$TASK_DATA_FOLDER/processed/$SPLIT.raw.$MYLANG" \
      --outputs "$TASK_DATA_FOLDER/processed/$SPLIT.$MYLANG" \
      --workers 60 \
      --keep-empty;
    done
  done

  # Remove output directory.
  rm -rf "$TASK-bin" ||:

  DEVPREF="$TASK_DATA_FOLDER/processed/val.LANG"
  TESTPREF="$TASK_DATA_FOLDER/processed/test.LANG"

  # Run fairseq preprocessing:
  for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
  do
    MYLANG="input$INPUT_TYPE"
    python ../../fairseq_cli/preprocess.py \
      --only-source \
      --trainpref "$TASK_DATA_FOLDER/processed/train.$MYLANG" \
      --validpref "${DEVPREF//LANG/$MYLANG}" \
      --testpref "${TESTPREF//LANG/$MYLANG}" \
      --destdir "${OUTPUT}/$TASK-bin/$MYLANG" \
      --workers 8 \
      --srcdict $DICT/dict.txt;
  done
  if [[ "$TASK" !=  "STS-B" ]]
  then
    python ../../fairseq_cli/preprocess.py \
      --only-source \
      --trainpref "$TASK_DATA_FOLDER/processed/train.label" \
      --validpref "${DEVPREF//LANG/'label'}" \
      --destdir "${OUTPUT}/$TASK-bin/label" \
      --workers 8;
  else
    # For STS-B output range is converted to be between: [0.0, 1.0]
    mkdir -p "${OUTPUT}/$TASK-bin/label"
    awk '{print $1 / 5.0 }' "$TASK_DATA_FOLDER/processed/train.label" > "${OUTPUT}/$TASK-bin/label/train.label"
    awk '{print $1 / 5.0 }' "$TASK_DATA_FOLDER/processed/dev.label" > "${OUTPUT}/$TASK-bin/label/valid.label"
  fi
done
