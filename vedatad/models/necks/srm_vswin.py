import torch.nn as nn

from vedacore.misc import build_from_module, registry


@registry.register_module('neck')
class SRMSwin(nn.Module):
    """Spatial Reduction Module."""

    def __init__(self, srm_cfg):
        super(SRMSwin, self).__init__()
        # self.srm = build_from_module(srm_cfg, nn)

        in_channels = srm_cfg['in_channels']
        out_channels = srm_cfg['out_channels']

        self.pooling = nn.AdaptiveAvgPool3d([None, 1, 1])
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def init_weights(self):
        pass

    def forward(self, x):
        x = self.pooling(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x
