# Copyright (c) Microsoft. All rights reserved.
import yaml
import os
import argparse
import json
from pytorch_pretrained_bert.tokenization import BertTokenizer
import sentencepiece as spm

from data_utils import load_data
from data_utils.task_def import TaskType, DataFormat
from data_utils.log_wrapper import create_logger
from experiments.exp_def import TaskDefs
from data_utils.xlnet_utils import preprocess_text, encode_ids
from data_utils.xlnet_utils import CLS_ID, SEP_ID

DEBUG_MODE=False
MAX_SEQ_LEN = 512

### XLNET ###
SEG_ID_A   = 0
SEG_ID_B   = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4
### XLNET ###

logger = create_logger(__name__, to_disk=True, log_file='bert_data_proc_{}.log'.format(MAX_SEQ_LEN))

def xlnet_tokenize_fn(text, sp):
    text = preprocess_text(text)
    return encode_ids(sp, text)

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

def xlnet_feature_extractor(text_a, text_b=None, max_seq_length=512, tokenize_fn=None):
    tokens_a = xlnet_tokenize_fn(text_a, tokenize_fn)
    tokens_b = None
    if text_b:
        tokens_b = xlnet_tokenize_fn(text_a, tokenize_fn)

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for one [SEP] & one [CLS] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]
    tokens = []
    segment_ids = []
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(SEG_ID_A)
    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_A)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(SEG_ID_B)
        tokens.append(SEP_ID)
        segment_ids.append(SEG_ID_B)

    tokens.append(CLS_ID)
    segment_ids.append(SEG_ID_CLS)
    input_ids = tokens

    # The mask has 0 for real tokens and 1 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    if len(input_ids) < max_seq_length:
        delta_len = max_seq_length - len(input_ids)
        input_ids = [0] * delta_len + input_ids
        input_mask = [1] * delta_len + input_mask
        segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids

def bert_feature_extractor(text_a, text_b=None, max_seq_length=512, tokenize_fn=None):
    tokens_a = tokenize_fn.tokenize(text_a)
    tokens_b = None
    if text_b:
        tokens_b = tokenize_fn.tokenize(text_b)

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for one [SEP] & one [CLS] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:max_seq_length - 2]
    if tokens_b:
        input_ids = tokenize_fn.convert_tokens_to_ids(['[CLS]'] + tokens_b + ['[SEP]'] + tokens_a + ['[SEP]'])
        segment_ids = [0] * ( len(tokens_b) + 2) + [1] * (len(tokens_a) + 1)
    else:
        input_ids = tokenize_fn.convert_tokens_to_ids(['[CLS]'] + tokens_a + ['[SEP]'])
        segment_ids = [0] * len(input_ids)
    input_mask = None
    return input_ids, input_mask, segment_ids


def build_data(data, dump_path, tokenizer, data_format=DataFormat.PremiseOnly, max_seq_len=MAX_SEQ_LEN, is_bert_model=True):
    def build_data_premise_only(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, is_bert_model=True):
        """Build data of single sentence tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                label = sample['label']
                if len(premise) >  max_seq_len - 2:
                    premise = premise[:max_seq_len - 2]
                if is_bert_model:
                    input_ids, _, type_ids = bert_feature_extractor(premise, max_seq_length=max_seq_len, tokenize_fn=tokenizer)
                    features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
                else:
                    input_ids, input_mask, type_ids = xlnet_feature_extractor(premise, max_seq_length=max_seq_len, tokenize_fn=tokenizer)
                    features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids, 'mask': input_mask}
                writer.write('{}\n'.format(json.dumps(features)))


    def build_data_premise_and_one_hypo(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, is_bert_model=True):
        """Build data of sentence pair tasks
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis = sample['hypothesis']
                label = sample['label']
                if is_bert_model:
                    input_ids, _, type_ids = bert_feature_extractor(premise, hypothesis, max_seq_length=max_seq_len, tokenize_fn=tokenizer)
                    features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}
                else:
                    input_ids, input_mask, type_ids = xlnet_feature_extractor(premise, hypothesis, max_seq_length=max_seq_len, tokenize_fn=tokenizer)
                    features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids, 'mask': input_mask}
                writer.write('{}\n'.format(json.dumps(features)))

    def build_data_premise_and_multi_hypo(data, dump_path, max_seq_len=MAX_SEQ_LEN, tokenizer=None, is_bert_model=True):
        """Build QNLI as a pair-wise ranking task
        """
        with open(dump_path, 'w', encoding='utf-8') as writer:
            for idx, sample in enumerate(data):
                ids = sample['uid']
                premise = sample['premise']
                hypothesis_1 = sample['hypothesis'][0]
                hypothesis_2 = sample['hypothesis'][1]
                label = sample['label']
                if is_bert_model:
                    input_ids_1, _, type_ids_1 = bert_feature_extractor(premise, hypothesis_1, max_seq_length=max_seq_len, tokenize_fn=tokenizer)
                    input_ids_2, _, type_ids_2 = bert_feature_extractor(premise, hypothesis_2, max_seq_length=max_seq_len, tokenize_fn=tokenizer)
                    features = {'uid': ids, 'label': label, 'token_id': [input_ids_1, input_ids_2], 'type_id': [type_ids_1, type_ids_2], 'ruid':sample['ruid'], 'olabel':sample['olabel']}
                else:
                    input_ids_1, mask_1, type_ids_1 = bert_feature_extractor(premise, hypothesis_1, max_seq_length=max_seq_len, tokenize_fn=tokenizer)
                    input_ids_2, mask_2, type_ids_2 = bert_feature_extractor(premise, hypothesis_2, max_seq_length=max_seq_len, tokenize_fn=tokenizer)
                    features = {'uid': ids, 'label': label, 'token_id': [input_ids_1, input_ids_2], 'type_id': [type_ids_1, type_ids_2], 'mask':[mask_1, mask_2], 'ruid':sample['ruid'], 'olabel':sample['olabel']}
                writer.write('{}\n'.format(json.dumps(features)))

    if data_format == DataFormat.PremiseOnly:
        build_data_premise_only(data, dump_path, max_seq_len, tokenizer, is_bert_model)
    elif data_format == DataFormat.PremiseAndOneHypothesis:
        build_data_premise_and_one_hypo(data, dump_path, max_seq_len, tokenizer, is_bert_model)
    elif data_format == DataFormat.PremiseAndMultiHypothesis:
        build_data_premise_and_multi_hypo(data, dump_path, max_seq_len, tokenizer, is_bert_model)
    else:
        raise ValueError(data_format)


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing GLUE/SNLI/SciTail dataset.')
    parser.add_argument('--model', type=str, default='bert-base-uncased', 
                        help='bert-base-uncased/bert-large-uncased/xlnet-large-cased')
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
    if 'uncased' in args.model:
        is_uncased = True

    is_bert_model = True
    if 'xlnet' in args.model:
        is_bert_model = False
    
    if is_bert_model:
        tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=do_lower_case)
    else:
        tokenizer = spm.SentencePieceProcessor()
        if 'large' in args.model:
            tokenizer.load('mt_dnn_models/xlnet_large_cased_spiece.model')
        else:
            tokenizer.load('mt_dnn_models/xlnet_base_cased_spiece.model')

    mt_dnn_suffix = 'mt_dnn_b' if is_bert_model else 'mt_dnn_x'
    if is_uncased:
        mt_dnn_suffix = '{}_uncased'.format(mt_dnn_suffix)
    else:
        mt_dnn_suffix = '{}_cased'.format(mt_dnn_suffix)

    if do_lower_case:
        mt_dnn_suffix = '{}_lower'.format(mt_dnn_suffix)

    mt_dnn_root = os.path.join(root, mt_dnn_suffix)
    if not os.path.isdir(mt_dnn_root):
        os.mkdir(mt_dnn_root)

    task_defs = TaskDefs(args.task_def)
    for task in task_defs.tasks:
        logger.info("Task %s" % task)
        data_format = task_defs.data_format_map[task]
        task_type = task_defs.task_type_map[task]
        label_mapper = task_defs.global_map.get(task, None)
        split_names = task_defs.split_names_map[task]
        for split_name in split_names:
            rows = load_data(os.path.join(root, "%s_%s.tsv" % (task, split_name)), data_format, task_type, label_mapper)
            dump_path = os.path.join(mt_dnn_root, "%s_%s.json" % (task, split_name))
            logger.info(dump_path)
            build_data(rows, dump_path, tokenizer, data_format, is_bert_model=is_bert_model)

if __name__ == '__main__':
    args = parse_args()
    main(args)
