# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum
class TaskType(IntEnum):
    Classification = 1
    Regression = 2
    Ranking = 3
    Span = 4 # squad v1
    SpanYN = 5 # squad v2
    SeqenceLabeling = 6
    MaskLM = 7
    SpanSeqenceLabeling = 8
    SeqenceGeneration = 9


class DataFormat(IntEnum):
    PremiseOnly = 1
    PremiseAndOneHypothesis = 2
    PremiseAndMultiHypothesis = 3
    MRC = 4
    Seqence = 5
    MLM = 6
    CLUE_CLASSIFICATION = 7
    CLUE_SPAN = 8
    CLUE_SEQ = 9
    CLUE_GEN = 10 # generation


class EncoderModelType(IntEnum):
    BERT = 1
    ROBERTA = 2
    XLNET = 3
    SAN = 4
    XLM = 5
    DEBERTA = 6
    ELECTRA = 7
    T5 = 8
    T5G = 9