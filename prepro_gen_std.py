# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
# Generative finetune
# by xiaodong liu

import yaml
from tqdm import tqdm
import os
import numpy as np
import argparse
import json
import sys
from data_utils import load_data
from data_utils.task_def import TaskType, DataFormat
from data_utils.log_wrapper import create_logger
from experiments.exp_def import TaskDefs
from data_utils.tokenizer_utils import create_tokenizer
from functools import partial
import multiprocessing
import transformers

DEBUG_MODE = False
MAX_SEQ_LEN = 512
DOC_STRIDE = 180
MAX_QUERY_LEN = 64
MRC_MAX_SEQ_LEN = 384
MRC_MAX_GEN_LEN = 5

logger = create_logger(
    __name__, to_disk=True, log_file="mt_dnn_data_proc_{}.log".format(MAX_SEQ_LEN)
)


def feature_extractor(tokenizer, text_a, text_b=None, max_length=512, do_padding=False):
    inputs = tokenizer(
        text_a,
        text_b,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding=do_padding,
    )
    input_ids = inputs["input_ids"]
    token_type_ids = (
        inputs["token_type_ids"] if "token_type_ids" in inputs else [0] * len(input_ids)
    )
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = inputs["attention_mask"]
    if do_padding:
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length
        )
        assert (
            len(attention_mask) == max_length
        ), "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert (
            len(token_type_ids) == max_length
        ), "Error with input length {} vs {}".format(len(token_type_ids), max_length)
    return input_ids, attention_mask, token_type_ids

def label_tokenize(tokenizer, text, max_label_length=MRC_MAX_GEN_LEN):
    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_label_length,
        truncation=True,
    )
    input_ids = inputs["input_ids"]
    if type(tokenizer) is transformers.models.t5.tokenization_t5_fast.T5TokenizerFast:
        input_ids = [tokenizer.pad_token_id] + input_ids +  [tokenizer.eos_token_id]
    else:
        input_ids = [tokenizer._convert_token_to_id("[CLS]")] + input_ids +  [tokenizer._convert_token_to_id("[SEP]")]

    return input_ids


def extract_feature_premise_only(sample, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
    """extract feature of single sentence tasks"""
    input_ids, input_mask, type_ids = feature_extractor(
        tokenizer, sample["premise"], max_length=max_seq_len
    )
    label = label_tokenize(tokenizer, sample["label"]) if type(sample["label"]) is str else sample["label"]
    feature = {
        "uid": sample["uid"],
        "label": label,
        "token_id": input_ids,
        "type_id": type_ids,
        "attention_mask": input_mask,
        "answer": sample["label"]
    }
    return feature


def extract_feature_premise_and_one_hypo(
    sample, max_seq_len=MAX_SEQ_LEN, tokenizer=None,
    max_label_len=MRC_MAX_GEN_LEN,
):
    input_ids, input_mask, type_ids = feature_extractor(
        tokenizer,
        sample["premise"],
        text_b=sample["hypothesis"],
        max_length=max_seq_len,
    )
    label = label_tokenize(tokenizer, sample["label"], max_label_length=max_label_len) if type(sample["label"]) is str else sample["label"]

    feature = {
        "uid": sample["uid"],
        "label": label,
        "token_id": input_ids,
        "type_id": type_ids,
        "attention_mask": input_mask,
        "answer": sample["label"]
    }
    return feature




def build_data(
    data,
    dump_path,
    tokenizer,
    data_format=DataFormat.PremiseOnly,
    max_seq_len=MAX_SEQ_LEN,
    max_label_len=MRC_MAX_GEN_LEN,
    lab_dict=None,
    do_padding=False,
    truncation=True,
    workers=1,
):
    if data_format == DataFormat.PremiseOnly:
        partial_feature = partial(
            extract_feature_premise_only, max_seq_len=max_seq_len, tokenizer=tokenizer
        )
    elif data_format == DataFormat.PremiseAndOneHypothesis:
        partial_feature = partial(
            extract_feature_premise_and_one_hypo,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
        )
    else:
        raise ValueError(data_format)

    if workers > 1:
        with multiprocessing.Pool(processes=workers) as pool:
            features = pool.map(partial_feature, data)
        logger.info("begin to write features")
        with open(dump_path, "w", encoding="utf-8") as writer:
            for feature in tqdm(features, total=len(features)):
                writer.write("{}\n".format(json.dumps(feature)))
    else:
        with open(dump_path, "w", encoding="utf-8") as writer:
            for sample in tqdm(data, total=len(data)):
                feature = partial_feature(sample)
                writer.write("{}\n".format(json.dumps(feature)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocessing NLU tasks as generation tasks."
    )
    parser.add_argument("--model",type=str, default="t5-base")
    parser.add_argument("--do_padding", action="store_true")
    parser.add_argument("--root_dir", type=str, default="data/canonical_data")
    parser.add_argument("--task_def", type=str, default="experiments/glue/glue_task_def.yml")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_label_len", type=int, default=10)
    parser.add_argument("--transformer_cache", default=".cache", type=str)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    return args


def main(args):
    # hyper param
    root = args.root_dir
    assert os.path.exists(root)
    tokenizer = create_tokenizer(args.model, args.transformer_cache)

    mt_dnn_root = os.path.join(root, args.model)
    if not os.path.isdir(mt_dnn_root):
        os.makedirs(mt_dnn_root)

    task_defs = TaskDefs(args.task_def)
    
    for task in task_defs.get_task_names():
        task_def = task_defs.get_task_def(task)
        logger.info("Task %s" % task)
        for split_name in task_def.split_names:
            file_path = os.path.join(root, "%s_%s.tsv" % (task, split_name))
            if not os.path.exists(file_path):
                logger.warning("File %s doesnot exit")
                sys.exit(1)
            rows = load_data(file_path, task_def, raw_label=True)
            dump_path = os.path.join(mt_dnn_root, "%s_%s.json" % (task, split_name))
            logger.info(dump_path)
            build_data(
                rows,
                dump_path,
                tokenizer,
                task_def.data_type,
                lab_dict=task_def.label_vocab,
                workers=args.workers,
                max_seq_len=args.max_seq_len
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
