## Quickstart

###Example of XNLI based on XLM-R
1. Download XNLI data </br>

2. Prepro </br>
   > python experiments\xnli\xnli_prepro.py </br>
   > python prepro_std.py --model xlm-roberta-base --task_def experiments\xnli\xnli_task_def.yml --rood_dir [XNLI-DIR]
3. Train
   > python train.py --data_dir data\canonical_data\xlm_base_cased\ --train_data xnli --test_data xnli --init_checkpoint xml-roberta-base --task_def experiments\xnli\xnli_task_def.yml --encoder_type 5
