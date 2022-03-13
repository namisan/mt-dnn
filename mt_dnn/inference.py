# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
import enum
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import insert
from data_utils.metrics import calc_metrics
from mt_dnn.batcher import Collater
from data_utils.task_def import TaskType
from data_utils.utils_qa import postprocess_qa_predictions
from copy import deepcopy
import numpy as np
import torch
from tqdm import tqdm


def extract_encoding(model, data, use_cuda=True):
    if use_cuda:
        model.cuda()
    sequence_outputs = []
    max_seq_len = 0
    for idx, (batch_info, batch_data) in enumerate(data):
        batch_info, batch_data = Collater.patch_data(use_cuda, batch_info, batch_data)
        sequence_output = model.encode(batch_info, batch_data)
        sequence_outputs.append(sequence_output)
        max_seq_len = max(max_seq_len, sequence_output.shape[1])

    new_sequence_outputs = []
    for sequence_output in sequence_outputs:
        new_sequence_output = torch.zeros(
            sequence_output.shape[0], max_seq_len, sequence_output.shape[2]
        )
        new_sequence_output[:, : sequence_output.shape[1], :] = sequence_output
        new_sequence_outputs.append(new_sequence_output)

    return torch.cat(new_sequence_outputs)


def reduce_multirc(uids, predictions, golds):
    assert len(uids) == len(predictions)
    assert len(uids) == len(golds)
    from collections import defaultdict

    predict_map = defaultdict(list)
    gold_map = defaultdict(list)
    for idx, uid in enumerate(uids):
        blocks = uid.split("_")
        assert len(blocks) == 3
        nuid = "_".join(blocks[:-1])
        predict_map[uid].append(predictions[idx])
        gold_map[uid].append(golds[idx])
    return predict_map, gold_map


def merge(src, tgt):
    def _mg(src, tgt):
        if isinstance(src, dict):
            for k, v in src.items():
                if k in tgt:
                    tgt[k] = _mg(v, tgt[k])
                else:
                    tgt[k] = v
        elif isinstance(src, list):
            tgt.extend(src)
        elif isinstance(src, tuple):
            if isinstance(src[0], list):
                for i, k in enumerate(src):
                    tgt[i].extend(src[i])
            else:
                tgt.extend(src)
        else:
            tgt = src
        return tgt

    if tgt is None or len(tgt) == 0:
        tgt = deepcopy(src)
        return tgt
    else:
        return _mg(src, tgt)


def eval_model(
    model,
    data,
    metric_meta,
    device,
    with_label=True,
    label_mapper=None,
    task_type=TaskType.Classification,
):
    predictions = []
    golds = []
    scores = []
    ids = []
    metrics = {}
    for (batch_info, batch_data) in tqdm(data, total=len(data)):
        batch_info, batch_data = Collater.patch_data(device, batch_info, batch_data)
        score, pred, gold = model.predict(batch_info, batch_data)
        scores = merge(score, scores)
        golds = merge(gold, golds)
        predictions = merge(pred, predictions)
        ids = merge(batch_info["uids"], ids)

    if task_type == TaskType.Span:
        predictions, golds = postprocess_qa_predictions(
            golds, scores, version_2_with_negative=False
        )
    elif task_type == TaskType.SpanYN:
        predictions, golds = postprocess_qa_predictions(
            golds, scores, version_2_with_negative=True
        )

    if with_label:
        metrics = calc_metrics(metric_meta, golds, predictions, scores, label_mapper)
    return metrics, predictions, scores, golds, ids
