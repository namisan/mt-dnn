import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from data_utils.task_def import TaskType
from tasks import MTDNNTask
from tasks import register_task

@register_task('Regression')
class RegressionTask(MTDNNTask):
    def __init__(self, task_def):
        super().__init__(task_def)

    def input_parse_label(self, label: str):
        return float(label)

    @staticmethod
    def train_prepare_label(labels):
        return torch.FloatTensor(labels)

    @staticmethod
    def train_prepare_soft_label(softlabels):
        return torch.FloatTensor(softlabels)

    @staticmethod
    def test_predict(score, **kwargs):
        score = score.data.cpu()
        score = score.numpy()
        predict = np.argmax(score, axis=1).tolist()
        score = score.reshape(-1).tolist()
        return score, predict, batch_meta['label'], batch_meta['uids']
