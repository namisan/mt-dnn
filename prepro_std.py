# Copyright (c) Microsoft. All rights reserved.
import yaml
import os
import numpy as np
import argparse
import json
from pytorch_pretrained_bert.tokenization import BertTokenizer

from data_utils.glue_utils import *
from data_utils.label_map import GLOBAL_MAP, TaskType
from data_utils.log_wrapper import create_logger
from data_utils.vocab import Vocabulary

DEBUG_MODE=False
MAX_SEQ_LEN = 512

logger = create_logger(__name__, to_disk=True, log_file='bert_data_proc_512.log')

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.
    Copyed from https://github.com/huggingface/pytorch-pretrained-BERT
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def build_data(data, dump_path, tokenizer, data_format=DataFormat.PremiseOnly, max_seq_len=MAX_SEQ_LEN):
    def build_data_premise_only(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
        """Build data of single sentence tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = tokenizer.tokenize(sample['premise'])
                label = sample['label']
                if len(premise) >  max_seq_len - 3:
                    premise = premise[:max_seq_len - 3]
                input_ids =tokenizer.convert_tokens_to_ids(['[CLS]'] + premise + ['[SEP]'])
                type_ids = [0] * ( len(premise) + 2)
                features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
                writer.write('{}\n'.format(json.dumps(features)))


    def build_data_premise_and_one_hypo(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
        """Build data of sentence pair tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = tokenizer.tokenize(sample['premise'])
                hypothesis = tokenizer.tokenize(sample['hypothesis'])
                label = sample['label']
                _truncate_seq_pair(premise, hypothesis, max_seq_len - 3)
                input_ids =tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis + ['[SEP]'] + premise + ['[SEP]'])
                type_ids = [0] * ( len(hypothesis) + 2) + [1] * (len(premise) + 1)
                features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_premise_and_multi_hypo(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None):
        """Build QNLI as a pair-wise ranking task
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = tokenizer.tokenize(sample['premise'])
                hypothesis_1 = tokenizer.tokenize(sample['hypothesis'][0])
                hypothesis_2 = tokenizer.tokenize(sample['hypothesis'][1])
                label = sample['label']
                _truncate_seq_pair(premise, hypothesis_1, max_seq_len - 3)
                input_ids_1 =tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis_1 + ['[SEP]'] + premise + ['[SEP]'])
                type_ids_1 = [0] * ( len(hypothesis_1) + 2) + [1] * (len(premise) + 1)
                _truncate_seq_pair(premise, hypothesis_2, max_seq_len - 3)
                input_ids_2 = tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis_2 + ['[SEP]'] + premise + ['[SEP]'])
                type_ids_2 = [0] * ( len(hypothesis_2) + 2) + [1] * (len(premise) + 1)
                features = {'uid': ids, 'label': label, 'token_id': [input_ids_1, input_ids_2], 'type_id': [type_ids_1, type_ids_2], 'ruid':sample['ruid'], 'olabel':sample['olabel']}
                writer.write('{}\n'.format(json.dumps(features)))

    if data_format == DataFormat.PremiseOnly:
        build_data_premise_only(data, dump_path, max_seq_len, tokenizer)
    elif data_format == DataFormat.PremiseAndOneHypothesis:
        build_data_premise_and_one_hypo(data, dump_path, max_seq_len, tokenizer)
    elif data_format == DataFormat.PremiseAndMultiHypothesis:
        build_data_premise_and_multi_hypo(data, dump_path, max_seq_len, tokenizer)
    else:
        raise ValueError(data_format)


def load_data(file_path, data_format, task_type, label_dict=None):
    """
    :param file_path:
    :param data_format:
    :param task_type:
    :param label_dict: map string label to numbers.
        only valid for Classification task or ranking task.
        For ranking task, better label should have large number
    :return:
    """
    if task_type == TaskType.Ranking:
        assert data_format == DataFormat.PremiseAndMultiHypothesis

    rows = []
    for line in open(file_path, encoding="utf-8"):
        fields = line.strip("\n").split("\t")
        if data_format == DataFormat.PremiseOnly:
            assert len(fields) == 3
            row = {"uid": fields[0], "label": fields[1], "premise": fields[2]}
        elif data_format == DataFormat.PremiseAndOneHypothesis:
            assert len(fields) == 4
            row = {"uid": fields[0], "label": fields[1], "premise": fields[2], "hypothesis": fields[3]}
        elif data_format == DataFormat.PremiseAndMultiHypothesis:
            assert len(fields) > 5
            row = {"uid": fields[0], "ruid": fields[1].split(","), "label": fields[2], "premise": fields[3],
                   "hypothesis": fields[4:]}
        else:
            raise ValueError(data_format)

        if task_type == TaskType.Classification:
            if label_dict is not None:
                row["label"] = label_dict[row["label"]]
            else:
                row["label"] = int(row["label"])
        elif task_type == TaskType.Regression:
            row["label"] = float(row["label"])
        elif task_type == TaskType.Ranking:
            labels = row["label"].split(",")
            if label_dict is not None:
                labels = [label_dict[label] for label in labels]
            else:
                labels = [float(label) for label in labels]
            row["label"] = int(np.argmax(labels))
            row["olabel"] = labels

        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased')
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--root_dir', type=str, default='data/canonical_data')
    parser.add_argument('--task_def', type=str, default="task_def.yml")
    args = parser.parse_args()
    return args


def main(args):
    ## hyper param
    do_lower_case = args.do_lower_case
    root = args.root_dir
    assert os.path.exists(root)
    is_uncased = False
    if 'uncased' in args.bert_model:
        is_uncased = True

    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=do_lower_case)

    mt_dnn_suffix = 'mt_dnn'
    if is_uncased:
        mt_dnn_suffix = '{}_uncased'.format(mt_dnn_suffix)
    else:
        mt_dnn_suffix = '{}_cased'.format(mt_dnn_suffix)

    if do_lower_case:
        mt_dnn_suffix = '{}_lower'.format(mt_dnn_suffix)

    mt_dnn_root = os.path.join(root, mt_dnn_suffix)
    if not os.path.isdir(mt_dnn_root):
        os.mkdir(mt_dnn_root)

    task_def_dic = yaml.safe_load(open(args.task_def))

    for task, task_def in task_def_dic.items():
        logger.info("Task %s" % task)
        data_format = DataFormat[task_def["data_format"]]
        task_type = TaskType[task_def["task_type"]]
        label_mapper = None
        if "labels" in task_def:
            labels = task_def["labels"]
            label_mapper = Vocabulary(True)
            for label in labels:
                label_mapper.add(label)
        split_names = task_def.get("split_names", ["train", "dev", "test"])
        for split_name in split_names:
            rows = load_data(os.path.join(root, "%s_%s.tsv" % (task, split_name)), data_format, task_type, label_mapper)
            dump_path = os.path.join(mt_dnn_root, "%s_%s.json" % (task, split_name))
            logger.info(dump_path)
            build_data(rows, dump_path, bert_tokenizer, data_format)

if __name__ == '__main__':
    args = parse_args()
    main(args)
