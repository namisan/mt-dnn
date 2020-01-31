#! /bin/sh
python experiments/glue/glue_prepro.py
python prepro_std.py --model bert-large-uncased --root_dir data/canonical_data --task_def experiments/glue/glue_task_def.yml --do_lower_case $1
