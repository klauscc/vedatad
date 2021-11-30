# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================
import os
import os.path as osp

import torch

ckpt_dir = "data/pretrained_models/vswin"
src_name = "swin_base_patch244_window877_kinetics400_22k.pth"
dst_name = "swin_base_patch244_window877_kinetics400_22k_keysfrom_backbone.pth"

src_path = osp.join(ckpt_dir, src_name)
dst_path = osp.join(ckpt_dir, dst_name)

state = torch.load(src_path)
state = state["state_dict"]
print(state.keys())
torch.save(state, dst_path)
