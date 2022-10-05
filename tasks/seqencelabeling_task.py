import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from data_utils.task_def import TaskType
from tasks import MTDNNTask
from tasks import register_task

@register_task('SeqenceLabeling')
class SeqenceLabelingTask(MTDNNTask):
    def __init__(self, task_def):
        super().__init__(task_def)

    def input_parse_label(self, label):
        label_dict = self._task_def.label_vocab
        if label_dict is not None:
            labels = [label_dict[l] for l in label]
        else:
            labels = [float(l) for l in label]
        return labels

    @staticmethod
    def train_prepare_label(labels):
        batch_size = len(labels)
        tok_len = max([len(lab) for lab in labels])
        tlab = torch.LongTensor(batch_size, tok_len).fill_(-1)
        for i, label in enumerate(labels):
            ll = len(label)
            tlab[i, :ll] = torch.LongTensor(label)
        return tlab

    @staticmethod
    def train_prepare_soft_label(softlabels):
        return torch.FloatTensor(softlabels)

    @staticmethod
    def train_build_task_layer(hidden_size, task_def=None, opt=None, prefix="sequence_label"):
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
        hidden_size = sequence_output.size(2)
        sequence_output = sequence_output.contiguous().view(-1, hidden_size)
        return task_layer(sequence_output)

    @staticmethod
    def test_predict(score, batch_meta):
        mask = batch_meta["mask"]
        score = score.contiguous()
        score = score.data.cpu()
        score = score.numpy()
        predict = np.argmax(score, axis=1).reshape(mask.size()).tolist()
        valied_lenght = mask.sum(1).tolist()
        final_predict = []
        for idx, p in enumerate(predict):
            final_predict.append(p[: valied_lenght[idx]])
        score = score.reshape(-1).tolist()
        return score, final_predict, batch_meta['label'], batch_meta['uids']