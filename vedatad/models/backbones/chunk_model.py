# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================


from vedatad.models.backbones.vswin import SwinTransformer3D
import torch
import torch.nn.functional as F
from vedatad.models.builder import build_backbone
from torch import nn
from vedacore.misc import registry


@registry.register_module("backbone")
class ChunkVideoSwin(SwinTransformer3D):
    """extract feature chunk wise"""

    def __init__(self, chunk_size, *args, **kwargs):
        super(ChunkVideoSwin, self).__init__(*args, **kwargs)
        self.chunk_size = chunk_size

    def forward(self, x):
        """
        Args:
            x (Tensor[B,C,D,H,W]): input.
        """
        B, C, D, H, W = x.shape
        chunk_size = self.chunk_size
        pad_d = 0
        if D % chunk_size != 0:
            pad_d = chunk_size - (D % chunk_size)
            D = D + pad_d
            x = F.pad(x, (0, 0, 0, 0, 0, pad_d))
        num_chunks = D // chunk_size
        x = (
            x.reshape(B, C, num_chunks, chunk_size, H, W)
            .permute(0, 2, 1, 3, 4, 5)
            .reshape(B * num_chunks, C, chunk_size, H, W)
        )
        x = super().forward(x)  # shape: [n, c, d, h, w]
        n, c, d, h, w = x.shape
        x = (
            x.reshape(B, num_chunks, c, d, h, w)
            .permute(0, 2, 1, 3, 4, 5)
            .reshape(B, c, num_chunks * d, h, w)  # shape: [B, c, D//2, h, w]
        )
        x = x[:, :, : num_chunks * d - pad_d // 2, :, :].contiguous()
        return x
