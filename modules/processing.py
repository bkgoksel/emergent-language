import torch.nn as nn

"""
    A Processing module takes an input from a stream and the independent memory
    of that stream and runs a single timestep of a GRU cell, followed by
    dropout and finally a linear ELU layer on top of the GRU output.
    It returns the output of the fully connected layer as well as the update to
    the independent memory.
"""
class ProcessingModule(nn.Module):
    def __init__(self, config):
        super(ProcessingModule, self).__init__()
        self.cell = nn.GRUCell(config.input_size, config.hidden_size)
        self.fully_connected = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ELU())

    def forward(self, x, m):
        m = self.cell(x, m)
        return self.fully_connected(m), m

