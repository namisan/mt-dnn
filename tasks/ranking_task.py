import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from data_utils.task_def import TaskType
from tasks import MTDNNTask
from tasks import register_task

@register_task('Ranking')
class RankingTask(MTDNNTask):
    def __init__(self, task_def):
        super().__init__(task_def)

    def input_parse_label(self, label: str):
        label_dict = self._task_def.label_vocab
        if label_dict is not None:
            return label_dict[label]
        else:
            return int(label)

    @staticmethod
    def train_prepare_label(labels):
        return torch.LongTensor(labels)

    @staticmethod
    def train_prepare_soft_label(softlabels):
        return torch.FloatTensor(softlabels)

    @staticmethod
    def prepare_input(batch):
        newbatch = []
        sizes = []
        for sample in batch:
            size = len(sample["token_id"])
            sizes.append(size)
            assert size == len(sample["type_id"])
            for idx in range(0, size):
                token_id = sample["token_id"][idx]
                type_id = sample["type_id"][idx]
                attention_mask = sample["attention_mask"][idx] if "attention_mask" in sample else [1] * len(token_id)
                uid = sample["ruid"][idx] if "ruid" in sample else sample["uid"]
                olab = sample["olabel"][idx]
                new_sample = deepcopy(sample)
                new_sample["uid"] = uid
                new_sample["token_id"] = token_id
                new_sample["type_id"] = type_id
                new_sample["attention_mask"] = attention_mask
                new_sample["true_label"] = olab
                newbatch.append(new_sample)
        return {"batch": newbatch, "chunk_sizes": sizes}

    @staticmethod
    def test_predict(score, batch_meta):
        score = F.softmax(score, dim=1)
        score = score.data.cpu()
        score = score.numpy()
        predict = np.argmax(score, axis=1).tolist()
        score = score.reshape(-1).tolist()
        return score, predict, batch_meta['label'], batch_meta['uids']
