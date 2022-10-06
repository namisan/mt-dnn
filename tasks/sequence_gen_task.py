import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from data_utils.task_def import TaskType
from tasks import MTDNNTask
from tasks import register_task

@register_task('SeqenceGeneration')
class SeqenceGenerationTask(MTDNNTask):
    def __init__(self, task_def):
        super().__init__(task_def)

    def input_parse_label(self, label):
        return label

    @staticmethod
    def train_prepare_label(batch, **kwargs):
        padding_token_id = kwargs["padding_token_id"]
        eos_token_id = kwargs["eos_token_id"]
        max_answer_seq_len = kwargs["max_answer_seq_len"]
        batch_size = len(batch)
        max_answer_seq_len = max(len(x["label"]) for x in batch)
        y_idxs = torch.LongTensor(batch_size, max_answer_seq_len).fill_(padding_token_id)
        label = torch.LongTensor(batch_size, max_answer_seq_len).fill_(padding_token_id)
        for i, sample in enumerate(batch):
            max_len = min(len(sample["label"]), max_answer_seq_len)
            local_label = sample["label"]
            if len(sample["label"]) > max_len:
                eos = local_label[-1]
                local_label = local_label[: max_len - 2] + [eos_token_id, padding_token_id]
            y_idxs[i][:max_len] = torch.LongTensor(local_label)
            label[i][:max_len] = torch.LongTensor(local_label[1:] + [padding_token_id])
        return y_idxs, label

    @staticmethod
    def train_prepare_soft_label(softlabels):
        return torch.FloatTensor(softlabels)

    @staticmethod
    def train_build_task_layer(hidden_size, task_def=None, opt=None, prefix="sequence_label"):
        return None
    
    # TODO redesign hypers
    @staticmethod
    def train_forward(sequence_output, premise_mask, hyp_mask, task_layer=None, enable_san=False):
        hidden_size = sequence_output.size(2)
        sequence_output = sequence_output.contiguous().view(-1, hidden_size)
        return sequence_output

    @staticmethod
    def test_prepare_label(batch_info, batch):
        batch_info["answer"] = [sample["answer"] for sample in batch]

    @staticmethod
    def test_predict(score, batch_meta, tokenizer=None):
        predicts = tokenizer.batch_decode(score, skip_special_tokens=True)
        predictions = []
        golds = []
        for idx, predict in enumerate(predicts):
            sample_id = batch_meta["uids"][idx]
            answer = batch_meta["answer"][idx]
            predict = predict.strip()
            predictions.append(predict)
            golds.append(answer)
        score = score.contiguous()
        score = score.data.cpu()
        score = score.numpy().tolist()
        return score, predictions, golds, batch_meta['uids']