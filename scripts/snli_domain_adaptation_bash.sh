# 2 v100
./domain_adaptation_run.sh snli_001_tl ../mt_dnn_models/mt_dnn_base.pt snli_001 snli ../data/domain_adaptation ../checkpoints 32 0,1 |tee snli_001_tl.log
./domain_adaptation_run.sh snli_01_tl ../mt_dnn_models/mt_dnn_base.pt snli_01 snli ../data/domain_adaptation ../checkpoints 32 0,1 |tee snli_01_tl.log
./domain_adaptation_run.sh snli_1_tl ../mt_dnn_models/mt_dnn_base.pt snli_1 snli ../data/domain_adaptation ../checkpoints 32 0,1 |tee snli_1_tl.log
./domain_adaptation_run.sh snli_full_tl ../mt_dnn_models/mt_dnn_base.pt snli snli ../data/domain_adaptation ../checkpoints 32 0,1 |tee snli_full_tl.log
