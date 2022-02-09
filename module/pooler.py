import torch.nn as nn
from module.common import activation
from module.dropout_wrapper import DropoutWrapper


class Pooler(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.1, actf="tanh"):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = activation(actf)
        self.dropout = DropoutWrapper(dropout_p=dropout_p)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        first_token_tensor = self.dropout(first_token_tensor)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
