# Copyright (c) Microsoft. All rights reserved.

SAN_META = {
    'mnli': 1,
    'snli': 1,
    'scitail': 1,
    'qqp': 1,
    'qnli': 1,
    'qnnli': 1,
    'wnli': 1,
    'rte': 1,
    'mrpc': 1,
    'diag': 0,
    'sst': 0,
    'stsb': 0,
    'cola': 0,
}

def generate_decoder_opt(task, max_opt):
    assert task in SAN_META
    opt_v = 0
    if SAN_META[task] and max_opt < 3:
        opt_v = max_opt
    return opt_v

from enum import Enum
class TaskType(Enum):
 Classification = 1
 Regression = 2
 Ranking = 3
