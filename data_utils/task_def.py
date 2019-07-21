# Copyright (c) Microsoft. All rights reserved.

from enum import Enum
class TaskType(Enum):
 Classification = 1
 Regression = 2
 Ranking = 3


class DataFormat(Enum):
    PremiseOnly = 1
    PremiseAndOneHypothesis = 2
    PremiseAndMultiHypothesis = 3