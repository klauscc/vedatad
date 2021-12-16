import torch.nn as nn

from vedacore.misc import build_from_module, registry
from vedatad.models.modules.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)


@registry.register_module("neck")
class SRMSwin(nn.Module):
    """Spatial Reduction Module."""

    def __init__(self, srm_cfg):
        super(SRMSwin, self).__init__()
        # self.srm = build_from_module(srm_cfg, nn)

        in_channels = srm_cfg["in_channels"]
        out_channels = srm_cfg["out_channels"]

        self.pooling = nn.AdaptiveAvgPool3d([None, 1, 1])
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

        self.with_transformer = False
        if hasattr(srm_cfg, "with_transformer") and srm_cfg["with_transformer"]:
            self.with_transformer = True
            transformer_encoder_cfg = srm_cfg["transformer"]
            attn_layer = TransformerEncoderLayer(
                **transformer_encoder_cfg["encoder_layer"]
            )
            self.encoder = TransformerEncoder(
                attn_layer, num_layers=transformer_encoder_cfg["num_layers"]
            )

    def init_weights(self):
        pass

    def forward(self, x):
        """
        Args:
            x (torch.Tensor) : video input. Shape: (B,C1,D1,H1,W1) or (B,C1,D1).

        Returns:
            torch.Tensor. Features of shape (B, C2, D2).
        """
        if x.dim() == 5:  # [B, C1, D1, H1, W1]
            x = self.pooling(x)
            x = x.squeeze(-1).squeeze(-1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)  # [B, C2, D2]
        if self.with_transformer:
            x = x.permute(2, 0, 1)  # [D2, B, C2]
            x = self.encoder(x)
            x = x.permute(1, 2, 0)  # [B, C2, D2]
        return x
