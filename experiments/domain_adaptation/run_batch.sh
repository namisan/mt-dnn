#!/usr/bin/env bash
############################### 
# Batch training script for domain adaptation.
# Xiaodong
############################### 


declare -a SCITAIL=('scitail_001' 'scitail_01' 'scitail_1' 'scitail')

## Scitail
for split in "${SCITAIL[@]}"
do
    export CUDA_VISIBLE_DEVICES=0
    if [ ${split} == "scitail_001" ]  || [ ${split} == "scitail_01" ]; then
        batch_size=8
    else
        batch_size=32
    fi
    bash experiments/domain_adaptation/run_domain_adaptation.sh data/domain_adaptation/ mt_dnn_models/mt_dnn_base_uncased.pt ${split} scitail ${batch_size}
done



declare -a SCITAIL=('snli_001' 'snli_01' 'snli_1' 'snli')

##SNLI
for split in "${SCITAIL[@]}"
do
    export CUDA_VISIBLE_DEVICES=0
    if [ ${split} == "snli_001" ]  || [ ${split} == "snli_01" ]; then
        batch_size=8
    else
        batch_size=32
    fi
    bash experiments/domain_adaptation/run_domain_adaptation.sh data/domain_adaptation/ mt_dnn_models/mt_dnn_base_uncased.pt ${split} snli 32
done


