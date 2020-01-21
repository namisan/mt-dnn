# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum
class TaskType(IntEnum):
    Classification = 1
    Regression = 2
    Ranking = 3
    Span = 4
    SeqenceLabeling = 5

class DataFormat(IntEnum):
    PremiseOnly = 1
    PremiseAndOneHypothesis = 2
    PremiseAndMultiHypothesis = 3
    MRC = 4
    Seqence = 5

class EncoderModelType(IntEnum):
    BERT = 1
    ROBERTA = 2
    XLNET = 3

    @classmethod
    def from_name(cls, name):
        for model_type, model_type_name in EncoderModelType.items():
            if model_type_name == name:
                return model_type
        raise ValueError('{} is not a valid model_type name'.format(name))