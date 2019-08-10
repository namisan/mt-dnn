# Copyright (c) Microsoft. All rights reserved.

from data_utils.vocab import Vocabulary
from data_utils.metrics import compute_acc, compute_f1, compute_mcc, compute_pearson, compute_spearman

# scitail
ScitailLabelMapper = Vocabulary(True)
ScitailLabelMapper.add('neutral')
ScitailLabelMapper.add('entails')

# label map
SNLI_LabelMapper = Vocabulary(True)
SNLI_LabelMapper.add('contradiction')
SNLI_LabelMapper.add('neutral')
SNLI_LabelMapper.add('entailment')

# qnli
QNLILabelMapper = Vocabulary(True)
QNLILabelMapper.add('not_entailment')
QNLILabelMapper.add('entailment')

GLOBAL_MAP = {
    'scitail': ScitailLabelMapper,
    'mnli': SNLI_LabelMapper,
    'snli': SNLI_LabelMapper,
    'rte': QNLILabelMapper,
    'diag': SNLI_LabelMapper,
}

# number of class
DATA_META = {
    'mnli': 3,
    'snli': 3,
    'rte': 2,
    'diag': 3,
}

DATA_TYPE = {
    'mnli': 0,
    'snli': 0,
    'rte': 0,
    'diag': 0,
}

DATA_SWAP = {
    'mnli': 0,
    'snli': 0,
    'rte': 0,
    'diag': 0,
}

# classification/regression
TASK_TYPE = {
    'mnli': 0,
    'snli': 0,
    'rte': 0,
    'diag': 0,
}

METRIC_META = {
    'mnli': [0],
    'snli': [0],
    'rte': [0],
    'diag': [0],
}

METRIC_NAME = {
    0: 'ACC',
    1: 'F1',
    2: 'MCC',
    3: 'Pearson',
    4: 'Spearman',
}

METRIC_FUNC = {
    0: compute_acc,
    1: compute_f1,
    2: compute_mcc,
    3: compute_pearson,
    4: compute_spearman,
}

SAN_META = {
    'mnli': 1,
    'snli': 1,
    'rte': 1,
    'diag': 0,
}


def generate_decoder_opt(task, max_opt):
    assert task in SAN_META
    opt_v = 0
    if SAN_META[task] and max_opt < 3:
        opt_v = max_opt
    return opt_v


from enum import Enum


class TaskType(Enum):
    Classification = 0
    Regression = 1
    Ranking = 2
