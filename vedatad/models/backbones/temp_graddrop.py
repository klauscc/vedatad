# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================


from functools import lru_cache
from vedatad.models.backbones.vswin import SwinTransformer3D
import numpy as np
import torch
import torch.nn.functional as F
from vedatad.models.builder import build_backbone
from torch import nn
from vedacore.misc import registry


class TemporalGradDrop(torch.autograd.Function):
    """gradient drop for backbone"""

    @staticmethod
    def forward(ctx, keep_indices, drop_indices, forward_fn, x, *params):
        """

        Args:
            keep_indices (list): The indices of chunks to keep gradients
            forward_fn (nn.Module.forward): forward function
            x (torch.Tensor): the input. Shape: [num_chunks, B, C, chunk_size, H, W]. D = num_chunks * chunk_size.
            *params (List of nn.Parameters): The parameters that need gradients.

        Returns: forward output.

        """
        x_w_grad = x[keep_indices]

        # save for backward
        ctx.keep_indices = keep_indices
        ctx.tensors = (x_w_grad.detach().clone(), params)
        ctx.forward_fn = forward_fn

        y = forward_fn(x)
        return y

    @staticmethod
    def backward(ctx, dy):
        """backward

        Args:
            dy (torch.Tensor): Gradients to dy. shape: (num_chunks,B,C1,D1,H1,W1)

        Returns: gradients to inputs of forward.

        """
        keep_indices = ctx.keep_indices
        x_w_grad, params = ctx.tensors
        forward_fn = ctx.forward_fn

        # forward inputs need grads.
        with torch.enable_grad():
            y_w_grad = forward_fn(x_w_grad)  # [num_chunks_keep, B, C', D',H',W']
        d_y_w_grad = dy[keep_indices]
        params_grad = torch.autograd.grad(y_w_grad, params, d_y_w_grad)
        del x_w_grad, ctx.keep_indices
        return (
            None,
            None,
            None,
            None,
        ) + params_grad


def generate_indices(num_chunks, keep_ratio, mode="uniform"):
    """generate indices of inputs that need and don't need gradients.

    Args:
        num_chunks (int): number of chunks
        keep_ratio (float): keep ratio.

    Returns: TODO

    """
    if mode == "uniform":
        keep_indices = (
            np.floor(np.linspace(0, num_chunks - 1, int(num_chunks * keep_ratio)))
            .astype(np.int64)
            .tolist()
        )
    elif mode == "random":
        keep_indices = np.random.choice(
            np.arange(num_chunks), size=int(num_chunks * keep_ratio), replace=False
        ).tolist()
        keep_indices = sorted(keep_indices)
    else:
        raise ValueError(f"generate_indices: mode:{mode} not implemented")

    drop_indices = []
    for i in range(num_chunks):
        if i not in keep_indices:
            drop_indices.append(i)

    return keep_indices, drop_indices


@registry.register_module("backbone")
class GradDropChunkVideoSwin(SwinTransformer3D):
    """chunk-wise video swin with partial feedback."""

    def __init__(self, keep_ratio, chunk_size, *args, **kwargs):
        super(GradDropChunkVideoSwin, self).__init__(*args, **kwargs)
        self.chunk_size = chunk_size
        self.keep_ratio = keep_ratio

        self.graddrop_op = TemporalGradDrop.apply

    def forward_fn(self, x):
        """forward function

        Args:
            x (torch.Tensor): input video. shape: (num_chunks,B,C,chunk_size,H,W)

        Returns: TODO

        """
        num_chunks, B, C, chunk_size, H, W = x.shape
        x = x.reshape(
            num_chunks * B, C, chunk_size, H, W
        )  # shape: [num_chunks*B, C,D,H,W]
        y = super().forward(x)  # shape: [num_chunks*B, C', D', H', W']
        _, C1, D1, H1, W1 = y.shape
        y = y.reshape(num_chunks, B, C1, D1, H1, W1)
        return y

    def gather_trainable_parameters(self):
        """gather the trainable parameters
        Returns: List of trainable parameters.

        """
        trainable_params = []
        for param in self.parameters():
            if param.requires_grad:
                trainable_params.append(param)
        return trainable_params

    def forward(self, x):
        """forward

        Args:
            x (torch.Tensor): input video. shape: (B,C,D,H,W)

        Returns: TODO

        """
        B, C, D, H, W = x.shape
        chunk_size = self.chunk_size
        assert D % chunk_size == 0, "D mod chunk_size must be 0."
        num_chunks = D // chunk_size

        # transpose input
        x = x.reshape(B, C, num_chunks, chunk_size, H, W).permute(
            2, 0, 1, 3, 4, 5
        )  # shape: (num_chunks, B, C, chunk_size, H, W)

        # generate indices
        keep_indices, drop_indices = generate_indices(num_chunks, self.keep_ratio)

        trainable_params = self.gather_trainable_parameters()
        y = self.graddrop_op(
            keep_indices, drop_indices, self.forward_fn, x, *trainable_params
        )  # shape: (num_chunks, B,C1, D1,H1,W1)

        num_chunks, B, C1, D1, H1, W1 = y.shape
        y = y.permute(1, 2, 0, 3, 4, 5).reshape(B, C1, num_chunks * D1, H1, W1)
        return y


@registry.register_module("backbone")
class GradDropChunkVideoSwinV2(SwinTransformer3D):
    """chunk-wise video swin with partial feedback."""

    def __init__(self, keep_ratio, chunk_size, *args, bp_idx_mode="uniform", **kwargs):
        super(GradDropChunkVideoSwinV2, self).__init__(*args, **kwargs)
        self.chunk_size = chunk_size
        self.keep_ratio = keep_ratio
        self.bp_idx_mode = bp_idx_mode

    def forward_fn(self, x):
        """forward function

        Args:
            x (torch.Tensor): input video. shape: (num_chunks,B,C,chunk_size,H,W)

        Returns: the extracted features. shape: (num_chunks, B, C1, D1, H1, W1)

        """
        num_chunks, B, C, chunk_size, H, W = x.shape
        x = x.reshape(
            num_chunks * B, C, chunk_size, H, W
        )  # shape: [num_chunks*B, C,D,H,W]
        y = super().forward(x)  # shape: [num_chunks*B, C', D', H', W']
        _, C1, D1, H1, W1 = y.shape
        y = y.reshape(num_chunks * B, C1, D1, H1, W1)
        y = torch.nn.functional.adaptive_avg_pool3d(y, [None, 1, 1])
        y = y.squeeze(-1).squeeze(-1)
        y = y.reshape(num_chunks, B, C1, D1)
        return y

    def forward(self, x):
        """forward

        Args:
            x (torch.Tensor): input video. shape: (B,C,D,H,W)

        Returns: The extracted features. shape: (B, C1, D1, H1, W1)

        """
        B, C, D, H, W = x.shape
        chunk_size = self.chunk_size
        assert D % chunk_size == 0, "D mod chunk_size must be 0."
        num_chunks = D // chunk_size

        # transpose input
        x = x.reshape(B, C, num_chunks, chunk_size, H, W).permute(
            2, 0, 1, 3, 4, 5
        )  # shape: (num_chunks, B, C, chunk_size, H, W)

        # generate indices
        keep_indices, drop_indices = generate_indices(
            num_chunks, self.keep_ratio, self.bp_idx_mode
        )

        # ----- Batch forward ----------
        # orig v2.
        def batch_forward():
            with torch.no_grad():
                y_wo_grad = self.forward_fn(x[drop_indices].contiguous())
            y_w_grad = self.forward_fn(x[keep_indices].contiguous())
            return y_wo_grad, y_w_grad

        # --------------------------------

        # ----- Split forward ----------
        # new v2.
        def split_forward():
            with torch.no_grad():
                y_wo_grad = []
                for idx in drop_indices:
                    y_wo_grad.append(self.forward_fn(x[idx : idx + 1].contiguous()))
                y_wo_grad = torch.cat(y_wo_grad, dim=0)
            y_w_grad = self.forward_fn(x[keep_indices].contiguous())
            return y_wo_grad, y_w_grad

        # ----- Split forward restore RNG----------
        # v3.
        def split_forward_restore_rng():
            cpu_rng_state = torch.get_rng_state()
            gpu_rng_state = torch.cuda.get_rng_state()
            with torch.no_grad():
                y_wo_grad = []
                for idx in drop_indices:
                    torch.set_rng_state(cpu_rng_state)
                    torch.cuda.set_rng_state(gpu_rng_state)
                    y_wo_grad.append(self.forward_fn(x[idx : idx + 1].contiguous()))
                y_wo_grad = torch.cat(y_wo_grad, dim=0)
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(gpu_rng_state)
            y_w_grad = self.forward_fn(x[keep_indices].contiguous())
            return y_wo_grad, y_w_grad

        forward_mode = "batch_forward"
        if forward_mode == "batch_forward":
            y_wo_grad, y_w_grad = batch_forward()
        elif forward_mode == "split_forward":
            y_wo_grad, y_w_grad = split_forward()
        elif forward_mode == "split_forward_restore_rng":
            y_wo_grad, y_w_grad = split_forward_restore_rng()
        else:
            raise ValueError(f"forward mode:{forward_mode} not supported")

        _, B, C1, D1 = y_w_grad.shape
        y = torch.zeros(num_chunks, B, C1, D1, device=x.device, dtype=x.dtype)
        y[keep_indices] = y_w_grad
        y[drop_indices] = y_wo_grad

        y = y.permute(1, 2, 0, 3).reshape(B, C1, num_chunks * D1)
        return y
