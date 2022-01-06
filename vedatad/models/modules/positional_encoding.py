##########################################################################
# We adopt the positional encoding method from PyTorch Turorial.
# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
##########################################################################
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, scale_pe = False):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.scale_pe = scale_pe
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x, padding=0):
        pe = self.pe[padding : padding + x.shape[0], :]
        if self.scale_pe:
            pe = pe * (1. / math.sqrt(self.d_model))
        x = x + pe 
        return self.dropout(x)
