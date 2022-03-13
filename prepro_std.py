# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import yaml
import os
import numpy as np
import argparse
import json
import sys
from data_utils import load_data
from data_utils.task_def import TaskType, DataFormat
from data_utils.log_wrapper import create_logger
from experiments.exp_def import TaskDefs
from transformers import AutoTokenizer
from tqdm import tqdm
from functools import partial
import multiprocessing


DEBUG_MODE = False
MAX_SEQ_LEN = 512
DOC_STRIDE = 180
MAX_QUERY_LEN = 64
MRC_MAX_SEQ_LEN = 384

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


def extract_feature_premise_only(sample, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
    """extract feature of single sentence tasks"""
    input_ids, input_mask, type_ids = feature_extractor(
        tokenizer, sample["premise"], max_length=max_seq_len
    )
    feature = {
        "uid": sample["uid"],
        "label": sample["label"],
        "token_id": input_ids,
        "type_id": type_ids,
        "attention_mask": input_mask,
    }
    return feature


def extract_feature_premise_and_one_hypo(
    sample, max_seq_len=MAX_SEQ_LEN, tokenizer=None
):
    input_ids, input_mask, type_ids = feature_extractor(
        tokenizer,
        sample["premise"],
        text_b=sample["hypothesis"],
        max_length=max_seq_len,
    )
    feature = {
        "uid": sample["uid"],
        "label": sample["label"],
        "token_id": input_ids,
        "type_id": type_ids,
        "attention_mask": input_mask,
    }
    return feature


def extract_feature_premise_and_multi_hypo(
    sample, max_seq_len=MAX_SEQ_LEN, tokenizer=None
):
    ids = sample["uid"]
    premise = sample["premise"]
    hypothesis_list = sample["hypothesis"]
    label = sample["label"]
    input_ids_list = []
    type_ids_list = []
    attention_mask_list = []
    for hypothesis in hypothesis_list:
        input_ids, input_mask, type_ids = feature_extractor(
            tokenizer, premise, hypothesis, max_length=max_seq_len
        )
        input_ids_list.append(input_ids)
        type_ids_list.append(type_ids)
        attention_mask_list.append(input_mask)
    feature = {
        "uid": ids,
        "label": label,
        "token_id": input_ids_list,
        "type_id": type_ids_list,
        "ruid": sample["ruid"],
        "olabel": sample["olabel"],
        "attention_mask": attention_mask_list,
    }
    return feature


def extract_feature_sequence(
    sample, max_seq_len=MAX_SEQ_LEN, tokenizer=None, label_mapper=None
):
    ids = sample["uid"]
    premise = sample["premise"]
    tokens = []
    labels = []
    for i, word in enumerate(premise):
        subwords = tokenizer.tokenize(word)
        tokens.extend(subwords)
        for j in range(len(subwords)):
            if j == 0:
                labels.append(sample["label"][i])
            else:
                labels.append(label_mapper["X"])
    if len(premise) > max_seq_len - 2:
        tokens = tokens[: max_seq_len - 2]
        labels = labels[: max_seq_len - 2]

    label = [label_mapper["CLS"]] + labels + [label_mapper["SEP"]]
    input_ids = tokenizer.convert_tokens_to_ids(
        [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    )
    assert len(label) == len(input_ids)
    type_ids = [0] * len(input_ids)
    feature = {"uid": ids, "label": label, "token_id": input_ids, "type_id": type_ids}
    return feature

def extract_feature_cloze_choice(
    sample, max_seq_len=MAX_SEQ_LEN, tokenizer=None
):
    ids = sample["uid"]
    premise = sample["premise"]
    hypothesis_list = sample["hypothesis"]
    label = sample["label"]
    input_ids_list = []
    type_ids_list = []
    attention_mask_list = []
    for hypothesis in hypothesis_list:
        input_ids, input_mask, type_ids = feature_extractor(
            tokenizer, premise, hypothesis, max_length=max_seq_len
        )
        input_ids_list.append(input_ids)
        type_ids_list.append(type_ids)
        attention_mask_list.append(input_mask)
    feature = {
        "uid": ids,
        "label": label,
        "token_id": input_ids_list,
        "type_id": type_ids_list,
        "olabel": sample["olabel"],
        "attention_mask": attention_mask_list,
        "choice": sample["choice"],
        "answer": sample["answer"]
    }
    return feature

def build_data(
    data,
    dump_path,
    tokenizer,
    data_format=DataFormat.PremiseOnly,
    max_seq_len=MAX_SEQ_LEN,
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
    elif data_format == DataFormat.PremiseAndMultiHypothesis:
        partial_feature = partial(
            extract_feature_premise_and_multi_hypo,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
        )
    elif data_format == DataFormat.Seqence:
        partial_feature = partial(
            extract_feature_sequence,
            max_seq_len=max_seq_len,
            tokenizer=tokenizer,
            label_mapper=lab_dict,
        )
    elif data_format == DataFormat.ClozeChoice:
        partial_feature = partial(
            extract_feature_cloze_choice,
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
        description="Preprocessing GLUE/SNLI/SciTail dataset."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bert-base-uncased",
        help="support all BERT and ROBERTA family supported by HuggingFace Transformers",
    )
    parser.add_argument("--do_padding", action="store_true")
    parser.add_argument("--root_dir", type=str, default="data/canonical_data")
    parser.add_argument(
        "--task_def", type=str, default="experiments/glue/glue_task_def.yml"
    )
    parser.add_argument("--transformer_cache", default=".cache", type=str)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    return args


def main(args):
    # hyper param
    root = args.root_dir
    assert os.path.exists(root)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.transformer_cache
    )

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
            rows = load_data(file_path, task_def)
            dump_path = os.path.join(mt_dnn_root, "%s_%s.json" % (task, split_name))
            logger.info(dump_path)
            build_data(
                rows,
                dump_path,
                tokenizer,
                task_def.data_type,
                lab_dict=task_def.label_vocab,
                workers=args.workers,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
