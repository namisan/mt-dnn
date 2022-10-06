import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from data_utils.task_def import TaskType
from tasks import MTDNNTask
from tasks import register_task

@register_task('SpanClassification')
class SpanClassificationTask(MTDNNTask):
    def __init__(self, task_def):
        super().__init__(task_def)

    def input_parse_label(self, label):
        labels = [int(l) for l in label]
        return labels

    @staticmethod
    def train_prepare_label(batch):
        starts, ends = [], []
        for sample in batch:
            label = sample["label"]
            assert len(label) >= 2
            starts.append(label[0])
            ends.append(label[1])
        return (torch.LongTensor(starts), torch.LongTensor(ends))

    @staticmethod
    def train_prepare_soft_label(softlabels):
        return torch.FloatTensor(softlabels)

    @staticmethod
    def train_build_task_layer(hidden_size, task_def=None, opt=None, prefix="span_classification"):
        assert task_def.n_class == 2
        if task_def.dropout_p > 0:
            task_layer = nn.Sequential(
                nn.Dropout(task_def.dropout_p),
                nn.Linear(hidden_size, task_def.n_class)
            )
        else:
            task_layer = nn.Linear(hidden_size, task_def.n_class)
        return task_layer
    
    # TODO redesign hypers
    @staticmethod
    def train_forward(sequence_output, premise_mask, hyp_mask, task_layer=None, enable_san=False):
        logits = task_layer(sequence_output)
        start_scores, end_scores = logits.split(1, dim=-1)
        start_scores = start_scores.squeeze(-1)
        end_scores = end_scores.squeeze(-1)
        return start_scores, end_scores

    @staticmethod
    def test_prepare_label(batch_info, batch):
        batch_info["offset_mapping"] = [
            sample["offset_mapping"] for sample in batch
        ]
        batch_info["token_is_max_context"] = [
            sample.get("token_is_max_context", None) for sample in batch
        ]
        batch_info["context"] = [sample["context"] for sample in batch]
        batch_info["answer"] = [sample["answer"] for sample in batch]
        batch_info["label"] = [
            sample["label"] if "label" in sample else None
            for sample in batch
        ]


    @staticmethod
    def test_predict(score, batch_meta):
        predictions = []
        features = []
        uids = []
        for idx, offset in enumerate(batch_meta["offset_mapping"]):
            token_is_max_context = (
                batch_meta["token_is_max_context"][idx]
                if batch_meta.get("token_is_max_context", None)
                else None
            )
            sample_id = batch_meta["uids"][idx]
            uids.append(sample_id)
            if "label" in batch_meta:
                feature = {
                    "offset_mapping": offset,
                    "token_is_max_context": token_is_max_context,
                    "uid": sample_id,
                    "context": batch_meta["context"][idx],
                    "answer": batch_meta["answer"][idx],
                    "label": batch_meta["label"][idx],
                }
            else:
                feature = {
                    "offset_mapping": offset,
                    "token_is_max_context": token_is_max_context,
                    "uid": sample_id,
                    "context": batch_meta["context"][idx],
                    "answer": batch_meta["answer"][idx],
                }
            if "null_ans_index" in batch_meta:
                feature["null_ans_index"] = batch_meta["null_ans_index"]
            features.append(feature)
        start, end = score
        start = start.contiguous()
        start = start.data.cpu()
        start = start.numpy().tolist()
        end = end.contiguous()
        end = end.data.cpu()
        end = end.numpy().tolist()
        return (start, end), predictions, features, uids

@register_task('SpanClassificationYN')
class SpanClassificationYNTask(SpanClassificationTask):
    def __init__(self, task_def):
        super().__init__(task_def)

    @staticmethod
    def train_prepare_label(batch):
        starts, ends, yns = [], [], []
        for sample in batch:
            label = sample["label"]
            assert len(label) == 3
            starts.append(label[0])
            ends.append(label[1])
            yns.append(label[2])
        return (torch.LongTensor(starts), torch.LongTensor(ends), torch.LongTensor(yns))
