# Copyright (c) Microsoft. All rights reserved.
import os
import json
import tqdm
import pickle
import re
import collections
import argparse
from sys import path
from data_utils.vocab import Vocabulary
from pytorch_pretrained_bert.tokenization import BertTokenizer
from data_utils.log_wrapper import create_logger
from data_utils.label_map import GLOBAL_MAP
from data_utils.glue_utils import *
DEBUG_MODE=False
MAX_SEQ_LEN = 512

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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

def build_data(data, dump_path, max_seq_len=MAX_SEQ_LEN, is_train=True, tolower=True):
    """Build data of sentence pair tasks
    """
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = bert_tokenizer.tokenize(sample['premise'])
            hypothesis = bert_tokenizer.tokenize(sample['hypothesis'])
            label = sample['label']
            _truncate_seq_pair(premise, hypothesis, max_seq_len - 3)
            input_ids =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis + ['[SEP]'] + premise + ['[SEP]'])
            type_ids = [0] * ( len(hypothesis) + 2) + [1] * (len(premise) + 1)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
            writer.write('{}\n'.format(json.dumps(features)))

def build_qnli(data, dump_path, max_seq_len=MAX_SEQ_LEN, is_train=True, tolower=True, gold_path=None):
    """Build QNLI as a pair-wise ranking task
    """
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = bert_tokenizer.tokenize(sample['premise'])
            hypothesis_1 = bert_tokenizer.tokenize(sample['hypothesis'][0])
            hypothesis_2 = bert_tokenizer.tokenize(sample['hypothesis'][1])
            label = sample['label']
            _truncate_seq_pair(premise, hypothesis_1, max_seq_len - 3)
            input_ids_1 =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis_1 + ['[SEP]'] + premise + ['[SEP]'])
            type_ids_1 = [0] * ( len(hypothesis_1) + 2) + [1] * (len(premise) + 1)
            _truncate_seq_pair(premise, hypothesis_2, max_seq_len - 3)
            input_ids_2 =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + hypothesis_2 + ['[SEP]'] + premise + ['[SEP]'])
            type_ids_2 = [0] * ( len(hypothesis_2) + 2) + [1] * (len(premise) + 1)
            features = {'uid': ids, 'label': label, 'token_id': [input_ids_1, input_ids_2], 'type_id': [type_ids_1, type_ids_2], 'ruid':sample['ruid'], 'olabel':sample['olabel']}
            writer.write('{}\n'.format(json.dumps(features)))

def build_data_single(data, dump_path, max_seq_len=MAX_SEQ_LEN):
    """Build data of single sentence tasks
    """
    with open(dump_path, 'w', encoding='utf-8') as writer:
        for idx, sample in enumerate(data):
            ids = sample['uid']
            premise = bert_tokenizer.tokenize(sample['premise'])
            label = sample['label']
            if len(premise) >  max_seq_len - 3:
                premise = premise[:max_seq_len - 3] 
            input_ids =bert_tokenizer.convert_tokens_to_ids(['[CLS]'] + premise + ['[SEP]'])
            type_ids = [0] * ( len(premise) + 2)
            features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
            writer.write('{}\n'.format(json.dumps(features)))

def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--root_dir', type=str, default='data')
    args = parser.parse_args()
    return args

def main(args):
    root = args.root_dir
    assert os.path.exists(root)

    ######################################
    # SNLI/SciTail Tasks
    ######################################
    scitail_train_path = os.path.join(root, 'SciTail/tsv_format/scitail_1.0_train.tsv')
    scitail_dev_path = os.path.join(root, 'SciTail/tsv_format/scitail_1.0_dev.tsv')
    scitail_test_path = os.path.join(root, 'SciTail/tsv_format/scitail_1.0_test.tsv')

    snli_train_path = os.path.join(root, 'SNLI/train.tsv')
    snli_dev_path = os.path.join(root, 'SNLI/dev.tsv')
    snli_test_path = os.path.join(root, 'SNLI/test.tsv')

    ######################################
    # GLUE tasks
    ######################################
    multi_train_path =  os.path.join(root, 'MNLI/train.tsv')
    multi_dev_matched_path = os.path.join(root, 'MNLI/dev_matched.tsv')
    multi_dev_mismatched_path = os.path.join(root, 'MNLI/dev_mismatched.tsv')
    multi_test_matched_path = os.path.join(root, 'MNLI/test_matched.tsv')
    multi_test_mismatched_path = os.path.join(root, 'MNLI/test_mismatched.tsv')

    mrpc_train_path = os.path.join(root, 'MRPC/train.tsv')
    mrpc_dev_path = os.path.join(root, 'MRPC/dev.tsv')
    mrpc_test_path = os.path.join(root, 'MRPC/test.tsv')

    qnli_train_path = os.path.join(root, 'QNLI/train.tsv')
    qnli_dev_path = os.path.join(root, 'QNLI/dev.tsv')
    qnli_test_path = os.path.join(root, 'QNLI/test.tsv')

    qqp_train_path = os.path.join(root, 'QQP/train.tsv')
    qqp_dev_path = os.path.join(root, 'QQP/dev.tsv')
    qqp_test_path = os.path.join(root, 'QQP/test.tsv')

    rte_train_path = os.path.join(root, 'RTE/train.tsv')
    rte_dev_path = os.path.join(root, 'RTE/dev.tsv')
    rte_test_path = os.path.join(root, 'RTE/test.tsv')

    wnli_train_path = os.path.join(root, 'WNLI/train.tsv')
    wnli_dev_path = os.path.join(root, 'WNLI/dev.tsv')
    wnli_test_path = os.path.join(root, 'WNLI/test.tsv')

    stsb_train_path = os.path.join(root, 'STS-B/train.tsv')
    stsb_dev_path = os.path.join(root, 'STS-B/dev.tsv')
    stsb_test_path = os.path.join(root, 'STS-B/test.tsv')

    sst_train_path = os.path.join(root, 'SST-2/train.tsv')
    sst_dev_path = os.path.join(root, 'SST-2/dev.tsv')
    sst_test_path = os.path.join(root, 'SST-2/test.tsv')

    cola_train_path = os.path.join(root, 'CoLA/train.tsv')
    cola_dev_path = os.path.join(root, 'CoLA/dev.tsv')
    cola_test_path = os.path.join(root, 'CoLA/test.tsv')
    
    ######################################
    # Loading DATA
    ######################################
    scitail_train_data = load_scitail(scitail_train_path, GLOBAL_MAP['scitail'])
    scitail_dev_data = load_scitail(scitail_dev_path, GLOBAL_MAP['scitail'])
    scitail_test_data = load_scitail(scitail_test_path, GLOBAL_MAP['scitail'])
    logger.info('Loaded {} SciTail train samples'.format(len(scitail_train_data)))
    logger.info('Loaded {} SciTail dev samples'.format(len(scitail_dev_data)))
    logger.info('Loaded {} SciTail test samples'.format(len(scitail_test_data)))

    snli_train_data = load_snli(snli_train_path, GLOBAL_MAP['snli'])
    snli_dev_data = load_snli(snli_dev_path, GLOBAL_MAP['snli'])
    snli_test_data = load_snli(snli_test_path, GLOBAL_MAP['snli'])
    logger.info('Loaded {} SNLI train samples'.format(len(snli_train_data)))
    logger.info('Loaded {} SNLI dev samples'.format(len(snli_dev_data)))
    logger.info('Loaded {} SNLI test samples'.format(len(snli_test_data)))

    multinli_train_data = load_mnli(multi_train_path, GLOBAL_MAP['snli'])
    multinli_matched_dev_data = load_mnli(multi_dev_matched_path, GLOBAL_MAP['snli'])
    multinli_mismatched_dev_data = load_mnli(multi_dev_mismatched_path, GLOBAL_MAP['snli'])
    multinli_matched_test_data = load_mnli(multi_test_matched_path, GLOBAL_MAP['snli'], is_train=False)
    multinli_mismatched_test_data = load_mnli(multi_test_mismatched_path, GLOBAL_MAP['snli'], is_train=False)

    logger.info('Loaded {} MNLI train samples'.format(len(multinli_train_data)))
    logger.info('Loaded {} MNLI matched dev samples'.format(len(multinli_matched_dev_data)))
    logger.info('Loaded {} MNLI mismatched dev samples'.format(len(multinli_mismatched_dev_data)))
    logger.info('Loaded {} MNLI matched test samples'.format(len(multinli_matched_test_data)))
    logger.info('Loaded {} MNLI mismatched test samples'.format(len(multinli_mismatched_test_data)))

    mrpc_train_data = load_mrpc(mrpc_train_path)
    mrpc_dev_data = load_mrpc(mrpc_dev_path)
    mrpc_test_data = load_mrpc(mrpc_test_path, is_train=False)
    logger.info('Loaded {} MRPC train samples'.format(len(mrpc_train_data)))
    logger.info('Loaded {} MRPC dev samples'.format(len(mrpc_dev_data)))
    logger.info('Loaded {} MRPC test samples'.format(len(mrpc_test_data)))

    qnli_train_data = load_qnli(qnli_train_path, GLOBAL_MAP['qnli'])
    qnli_dev_data = load_qnli(qnli_dev_path, GLOBAL_MAP['qnli'])
    qnli_test_data = load_qnli(qnli_test_path, GLOBAL_MAP['qnli'], is_train=False)
    logger.info('Loaded {} QNLI train samples'.format(len(qnli_train_data)))
    logger.info('Loaded {} QNLI dev samples'.format(len(qnli_dev_data)))
    logger.info('Loaded {} QNLI test samples'.format(len(qnli_test_data)))
    
    qnnli_train_data = load_qnnli(qnli_train_path, GLOBAL_MAP['qnli'])
    qnnli_dev_data = load_qnnli(qnli_dev_path, GLOBAL_MAP['qnli'])
    qnnli_test_data = load_qnnli(qnli_test_path, GLOBAL_MAP['qnli'], is_train=False)
    logger.info('Loaded {} QNLI train samples'.format(len(qnli_train_data)))
    logger.info('Loaded {} QNLI dev samples'.format(len(qnli_dev_data)))
    logger.info('Loaded {} QNLI test samples'.format(len(qnli_test_data)))

    qqp_train_data = load_qqp(qqp_train_path)
    qqp_dev_data = load_qqp(qqp_dev_path)
    qqp_test_data = load_qqp(qqp_test_path, is_train=False)
    logger.info('Loaded {} QQP train samples'.format(len(qqp_train_data)))
    logger.info('Loaded {} QQP dev samples'.format(len(qqp_dev_data)))
    logger.info('Loaded {} QQP test samples'.format(len(qqp_test_data)))

    rte_train_data = load_rte(rte_train_path, GLOBAL_MAP['rte'])
    rte_dev_data = load_rte(rte_dev_path, GLOBAL_MAP['rte'])
    rte_test_data = load_rte(rte_test_path, GLOBAL_MAP['rte'], is_train=False)
    logger.info('Loaded {} RTE train samples'.format(len(rte_train_data)))
    logger.info('Loaded {} RTE dev samples'.format(len(rte_dev_data)))
    logger.info('Loaded {} RTE test samples'.format(len(rte_test_data)))

    wnli_train_data = load_wnli(wnli_train_path)
    wnli_dev_data = load_wnli(wnli_dev_path)
    wnli_test_data = load_wnli(wnli_test_path, is_train=False)
    logger.info('Loaded {} WNLI train samples'.format(len(wnli_train_data)))
    logger.info('Loaded {} WNLI dev samples'.format(len(wnli_dev_data)))
    logger.info('Loaded {} WNLI test samples'.format(len(wnli_test_data)))

    sst_train_data = load_sst(sst_train_path)
    sst_dev_data = load_sst(sst_dev_path)
    sst_test_data = load_sst(sst_test_path, is_train=False)
    logger.info('Loaded {} SST train samples'.format(len(sst_train_data)))
    logger.info('Loaded {} SST dev samples'.format(len(sst_dev_data)))
    logger.info('Loaded {} SST test samples'.format(len(sst_test_data)))

    stsb_train_data = load_sts(stsb_train_path)
    stsb_dev_data = load_sts(stsb_dev_path)
    stsb_test_data = load_sts(stsb_test_path, is_train=False)
    logger.info('Loaded {} STS-B train samples'.format(len(stsb_train_data)))
    logger.info('Loaded {} STS-B dev samples'.format(len(stsb_dev_data)))
    logger.info('Loaded {} STS-B test samples'.format(len(stsb_test_data)))

    cola_train_data = load_cola(cola_train_path)
    cola_dev_data = load_cola(cola_dev_path)
    cola_test_data = load_cola(cola_test_path, is_train=False)
    logger.info('Loaded {} COLA train samples'.format(len(cola_train_data)))
    logger.info('Loaded {} COLA dev samples'.format(len(cola_dev_data)))
    logger.info('Loaded {} COLA test samples'.format(len(cola_test_data)))

    mt_dnn_root = os.path.join(root, 'mt_dnn')
    if not os.path.isdir(mt_dnn_root):
        os.mkdir(mt_dnn_root)

    # BUILD SciTail
    scitail_train_fout = os.path.join(mt_dnn_root, 'scitail_train.json')
    scitail_dev_fout = os.path.join(mt_dnn_root, 'scitail_dev.json')
    scitail_test_fout = os.path.join(mt_dnn_root, 'scitail_test.json')
    build_data(scitail_train_data, scitail_train_fout)
    build_data(scitail_dev_data, scitail_dev_fout)
    build_data(scitail_test_data, scitail_test_fout)
    logger.info('done with scitail')

    # BUILD SNLI
    snli_train_fout = os.path.join(mt_dnn_root, 'snli_train.json')
    snli_dev_fout = os.path.join(mt_dnn_root, 'snli_dev.json')
    snli_test_fout = os.path.join(mt_dnn_root, 'snli_test.json')
    build_data(snli_train_data, snli_train_fout)
    build_data(snli_dev_data, snli_dev_fout)
    build_data(snli_test_data, snli_test_fout)
    logger.info('done with snli')

    # BUILD MNLI
    multinli_train_fout = os.path.join(mt_dnn_root, 'mnli_train.json')
    multinli_matched_dev_fout = os.path.join(mt_dnn_root, 'mnli_matched_dev.json')
    multinli_mismatched_dev_fout = os.path.join(mt_dnn_root, 'mnli_mismatched_dev.json')
    multinli_matched_test_fout = os.path.join(mt_dnn_root, 'mnli_matched_test.json')
    multinli_mismatched_test_fout = os.path.join(mt_dnn_root, 'mnli_mismatched_test.json')
    build_data(multinli_train_data, multinli_train_fout)
    build_data(multinli_matched_dev_data, multinli_matched_dev_fout)
    build_data(multinli_mismatched_dev_data, multinli_mismatched_dev_fout)
    build_data(multinli_matched_test_data, multinli_matched_test_fout)
    build_data(multinli_mismatched_test_data, multinli_mismatched_test_fout)
    logger.info('done with mnli')

    mrpc_train_fout = os.path.join(mt_dnn_root, 'mrpc_train.json')
    mrpc_dev_fout = os.path.join(mt_dnn_root, 'mrpc_dev.json')
    mrpc_test_fout = os.path.join(mt_dnn_root, 'mrpc_test.json')
    build_data(mrpc_train_data, mrpc_train_fout)
    build_data(mrpc_dev_data, mrpc_dev_fout)
    build_data(mrpc_test_data, mrpc_test_fout)
    logger.info('done with mrpc')

    qnli_train_fout = os.path.join(mt_dnn_root, 'qnli_train.json')
    qnli_dev_fout = os.path.join(mt_dnn_root, 'qnli_dev.json')
    qnli_test_fout = os.path.join(mt_dnn_root, 'qnli_test.json')
    build_data(qnli_train_data, qnli_train_fout)
    build_data(qnli_dev_data, qnli_dev_fout)
    build_data(qnli_test_data, qnli_test_fout)
    logger.info('done with qnli')

    qnli_train_fout = os.path.join(mt_dnn_root, 'qnnli_train.json')
    qnli_dev_fout = os.path.join(mt_dnn_root, 'qnnli_dev.json')
    qnli_test_fout = os.path.join(mt_dnn_root, 'qnnli_test.json')
    qnli_dev_gold_fout = os.path.join(mt_dnn_root, 'qnli_gold_dev.tsv')
    build_qnli(qnnli_train_data, qnli_train_fout)
    build_qnli(qnnli_dev_data, qnli_dev_fout)
    build_qnli(qnnli_train_data, qnli_test_fout)
    logger.info('done with qnli')

    qqp_train_fout = os.path.join(mt_dnn_root, 'qqp_train.json')
    qqp_dev_fout = os.path.join(mt_dnn_root, 'qqp_dev.json')
    qqp_test_fout = os.path.join(mt_dnn_root, 'qqp_test.json')
    build_data(qqp_train_data, qqp_train_fout)
    build_data(qqp_dev_data, qqp_dev_fout)
    build_data(qqp_test_data, qqp_test_fout)
    logger.info('done with qqp')

    rte_train_fout = os.path.join(mt_dnn_root, 'rte_train.json')
    rte_dev_fout = os.path.join(mt_dnn_root, 'rte_dev.json')
    rte_test_fout = os.path.join(mt_dnn_root, 'rte_test.json')
    build_data(rte_train_data, rte_train_fout)
    build_data(rte_dev_data, rte_dev_fout)
    build_data(rte_test_data, rte_test_fout)
    logger.info('done with rte')

    wnli_train_fout = os.path.join(mt_dnn_root, 'wnli_train.json')
    wnli_dev_fout = os.path.join(mt_dnn_root, 'wnli_dev.json')
    wnli_test_fout = os.path.join(mt_dnn_root, 'wnli_test.json')
    build_data(wnli_train_data, wnli_train_fout)
    build_data(wnli_dev_data, wnli_dev_fout)
    build_data(wnli_test_data, wnli_test_fout)
    logger.info('done with wnli')

    stsb_train_fout = os.path.join(mt_dnn_root, 'stsb_train.json')
    stsb_dev_fout = os.path.join(mt_dnn_root, 'stsb_dev.json')
    stsb_test_fout = os.path.join(mt_dnn_root, 'stsb_test.json')
    build_data(stsb_train_data, stsb_train_fout)
    build_data(stsb_dev_data, stsb_dev_fout)
    build_data(stsb_test_data, stsb_test_fout)
    logger.info('done with stsb')

    sst_train_fout = os.path.join(mt_dnn_root, 'sst_train.json')
    sst_dev_fout = os.path.join(mt_dnn_root, 'sst_dev.json')
    sst_test_fout = os.path.join(mt_dnn_root, 'sst_test.json')
    build_data_single(sst_train_data, sst_train_fout)
    build_data_single(sst_dev_data, sst_dev_fout)
    build_data_single(sst_test_data, sst_test_fout)
    logger.info('done with sst')

    cola_train_fout = os.path.join(mt_dnn_root, 'cola_train.json')
    cola_dev_fout = os.path.join(mt_dnn_root, 'cola_dev.json')
    cola_test_fout = os.path.join(mt_dnn_root, 'cola_test.json')
    build_data_single(cola_train_data, cola_train_fout)
    build_data_single(cola_dev_data, cola_dev_fout)
    build_data_single(cola_test_data, cola_test_fout)
    logger.info('done with cola')

if __name__ == '__main__':
    args = parse_args()
    main(args)
