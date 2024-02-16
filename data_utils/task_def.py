# Copyright (c) Microsoft. All rights reserved.

from enum import IntEnum


class TaskType(IntEnum):
    Classification = 1
    Regression = 2
    Ranking = 3
    SpanClassification = 4  # squad v1
    SpanClassificationYN = 5  # squad v2
    SeqenceLabeling = 6
    MaskLM = 7
    SpanSeqenceLabeling = 8
    SeqenceGeneration = 9
    SeqenceGenerationMRC = 10
    EncSeqenceGeneration = 11
    ClozeChoice = 12


class DataFormat(IntEnum):
    PremiseOnly = 1
    PremiseAndOneHypothesis = 2
    PremiseAndMultiHypothesis = 3
    Seqence = 4
    MLM = 5
    CLUE_CLASSIFICATION = 6
    CLUE_SPAN = 7
    CLUE_SEQ = 8
    CLUE_GEN = 9  # generation
    ClozeChoice = 10 #


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
    MSRT5G = 10
    MSRT5 = 11
    MSRLONGT5G = 12
    MSRLONGT5 = 13
    MISTRAL = 14
    MIXTRAL = 15
