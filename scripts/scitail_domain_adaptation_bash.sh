# 2 v100
./domain_adaptation_run.sh scitail_001_tl ../mt_dnn_models/mt_dnn_base.pt scitail_001 scitail ../data/domain_adaptation ../checkpoints 32 0,1 |tee scitail_001_tl.log
./domain_adaptation_run.sh scitail_01_tl ../mt_dnn_models/mt_dnn_base.pt scitail_01 scitail ../data/domain_adaptation ../checkpoints 32 0,1 |tee scitail_01_tl.log
./domain_adaptation_run.sh scitail_1_tl ../mt_dnn_models/mt_dnn_base.pt scitail_1 scitail ../data/domain_adaptation ../checkpoints 32 0,1 |tee scitail_1_tl.log
./domain_adaptation_run.sh scitail_full_tl ../mt_dnn_models/mt_dnn_base.pt scitail scitail ../data/domain_adaptation ../checkpoints 32 0,1 |tee scitail_full_tl.log
