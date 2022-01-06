#!/usr/bin/env python

import math
import torch
import torch.nn as nn

from vedacore.modules.bricks.conv_module import ConvModule
from vedatad.models.modules.positional_encoding import PositionalEncoding
from vedatad.models.modules.transformer import (
    TransformerDecoder,
    TransformerDecoderLayer,
)

from ..builder import build_neck
from vedacore.misc import registry


@registry.register_module("neck")
class AttnFPN(nn.Module):

    """FPN with skip connection"""

    def __init__(self, in_channels, out_channels, num_layers, neck_module_cfg):
        """TODO: to be defined.

        Args:
            in_channels (int): The input channels.
            out_channels (int): The out_channels of FPN.
            neck_module_cfg (list or dict): The neck module config.

        """
        super(AttnFPN, self).__init__()

        self.out_channels = out_channels
        self.neck_module_cfg = neck_module_cfg

        self.neck = build_neck(neck_module_cfg)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=None,
            act_cfg=dict(typename="ReLU"),
        )

        self.pe = PositionalEncoding(out_channels, scale_pe=True)

        decoder_layer = TransformerDecoderLayer(
            out_channels, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu"
        )
        self.trans_decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

    def init_weights(self):
        if isinstance(self.neck, nn.Sequential):
            for m in self.neck:
                m.init_weights()
        else:
            self.neck.init_weights()

    def forward(self, x):
        """forward function

        Args:
            x (torch.Tensor): the features. shape: (B, C, T)

        Returns: tuple. The FPN features. Each element is a tensor of shape (B, C', T'). T' is different for different levels.

        """
        pyramid_features = self.neck(x)
        high_res_feat = self.conv(x).permute(2, 0, 1)  # shape: [T1,B,C]
        outs = []
        for f in pyramid_features:  # shape: [B,C,T]
            f = f.permute(2, 0, 1)  # shape: [T,B,C]
            f = self.pe(f)
            f = self.trans_decoder(f, high_res_feat)
            f = f.permute(1, 2, 0)  # shape: [B,C,T]
            outs.append(f)
        return outs
