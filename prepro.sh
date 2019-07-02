#! /bin/sh
python glue_prepro.py
python prepro_std.py --bert_model bert-base-uncased --do_lower_case $1