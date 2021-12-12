import torch
import torch.nn as nn

from vedacore.misc import registry
from ..builder import build_backbone, build_head, build_neck
from .base_detector import BaseDetector

from vedatad.partial_feedback import indice_selection


@registry.register_module("detector")
class MemSingleStageDetector(BaseDetector):
    def __init__(self, chunk_size, backbone, head, neck=None):
        super().__init__()
        self.chunk_size = chunk_size
        self.backbone = build_backbone(backbone)
        if neck:
            self.neck = build_neck(neck)
        else:
            self.neck = None
        self.head = build_head(head)

        self.init_weights()

    def init_weights(self):
        self.backbone.init_weights()
        if self.neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

        self.head.init_weights()

    def forward_impl(self, x, frozen_features, keep_indices, drop_indices):
        """
        Args:
            x (torch.Tensor[num_chunks,B,C,chunk_size,H,W]): The imgs for bp chunks.
            frozen_features (torch.Tensor[num_nobp_chunks, B,C,feat_chunk_size]): The features for non-bp chunks.

        Returns: (feats, bp_features). `feat` is the final features by the entire model. `bp_features` is the features of bp-chunks from the backbone.
        """
        # extract the features for bp-chunks
        num_bp_chunks = x.shape[0]
        num_nobp_chunks = frozen_features.shape[0]
        num_chunks = num_bp_chunks + num_nobp_chunks
        bp_features = self.backbone(x)  # [num_keep_chunks, B, C, feat_chunk_size]

        # compose features
        feats = torch.zeros(
            [num_chunks] + list(bp_features.shape[1:]),
            dtype=bp_features.dtype,
            device=bp_features.device,
        )
        feats[keep_indices] = bp_features
        feats[drop_indices] = frozen_features
        ## [num_chunks, B, C, feat_chunk_size] -> [B, C, num_chunks*feat_chunk_size]
        num_chunks, B, C, feat_chunk_size = feats.shape
        feats = feats.permute(1, 2, 0, 3).reshape(B, C, num_chunks * feat_chunk_size)

        if self.neck:
            feats = self.neck(feats)
        feats = self.head(feats)
        return feats, bp_features

    def forward(self, x, frozen_features, keep_indices, drop_indices, train=True):
        if train:
            self.train()
            feats = self.forward_impl(x, frozen_features, keep_indices, drop_indices)
        else:
            self.eval()
            with torch.no_grad():
                feats = self.forward_impl(
                    x, frozen_features, keep_indices, drop_indices
                )
        return feats
