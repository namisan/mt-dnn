import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_utils.task_def import TaskType
from module.san import SANClassifier

TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()

class MTDNNTask:
    def __init__(self, task_def):
        self._task_def = task_def

    def input_parse_label(self, label: str):
        raise NotImplementedError()

    @staticmethod
    def input_is_valid_sample(sample, max_len):
         return len(sample['token_id']) <= max_len 
        

    @staticmethod
    def train_prepare_label(labels):
        raise NotImplementedError()

    
    @staticmethod
    def train_prepare_soft_label(softlabels):
        raise NotImplementedError()

    @staticmethod
    def train_build_task_layer(decoder_opt, hidden_size, lab, opt, prefix, dropout):
        if decoder_opt == 1:
            out_proj = SANClassifier(hidden_size, hidden_size, lab, opt, prefix, dropout=dropout)
        else:
            out_proj = nn.Linear(hidden_size, lab)
        return out_proj
    
    # TODO redesign hypers
    @staticmethod
    def train_forward(sequence_output, pooled_output, premise_mask, hyp_mask, decoder_opt, dropout_layer, task_layer):
        if decoder_opt == 1:
            max_query = hyp_mask.size(1)
            assert max_query > 0
            assert premise_mask is not None
            assert hyp_mask is not None
            hyp_mem = sequence_output[:, :max_query, :]
            logits = task_layer(sequence_output, hyp_mem, premise_mask, hyp_mask)
        else:
            pooled_output = dropout_layer(pooled_output)
            logits = task_layer(pooled_output)
        return logits
    
    @staticmethod
    def test_prepare_label(batch_info, labels):
        batch_info['label'] = labels
    
    @staticmethod
    def test_predict(score):
        raise NotImplementedError()


def register_task(name):
    """
        @register_task('Classification')
        class ClassificationTask(MTDNNTask):
            (...)

    .. note::

        All Tasks must implement the :class:`~MTDNNTask`
        interface.

    Args:
        name (str): the name of the task
    """

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            raise ValueError('Cannot register duplicate task ({})'.format(name))
        if not issubclass(cls, MTDNNTask):
            raise ValueError('Task ({}: {}) must extend MTDNNTask'.format(name, cls.__name__))
        if cls.__name__ in TASK_CLASS_NAMES:
            raise ValueError('Cannot register task with duplicate class name ({})'.format(cls.__name__))
        TASK_REGISTRY[name] = cls
        TASK_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_task_cls

def get_task_obj(task_def):
    task_name = task_def.task_type.name
    task_cls = TASK_REGISTRY.get(task_name, None)
    if task_cls is None:
        return None
    
    return task_cls(task_def)

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
    def test_predict(score):
        score = score.data.cpu()
        score = score.numpy()
        predict = np.argmax(score, axis=1).tolist()
        score = score.reshape(-1).tolist()
        return score, predict

@register_task('Classification')
class ClassificationTask(MTDNNTask):
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
    def test_predict(score):
        score = F.softmax(score, dim=1)
        score = score.data.cpu()
        score = score.numpy()
        predict = np.argmax(score, axis=1).tolist()
        score = score.reshape(-1).tolist()
        return score, predict

# TODO
# Span/SpanYN/SeqenceLabeling/SeqenceGeneration