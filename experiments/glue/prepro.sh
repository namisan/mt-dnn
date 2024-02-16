#!/usr/bin/env bash

############################### 
# Data prepro pipeline for MT-DNN.
# By xiaodong
############################### 

## dump original data into tsv
python experiments/glue/glue_prepro.py


declare -a PLMS=('bert-base-uncased' 'roberta-base' 'microsoft/deberta-base' 't5-base', 'mistralai/Mistral-7B-v0.1')

# prepro GLUE data for all PLMs.
for plm in "${PLMS[@]}"
do
  echo "Prepro GLUE data for $plm"
  echo "python prepro_std.py --model $plm --root_dir data/canonical_data --task_def experiments/glue/glue_task_def.yml --workers 32"
done


