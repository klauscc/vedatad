# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================

import json
import os
import time
import numpy as np
import torch
from vedatad.datasets.pipelines.formating import to_tensor
from vedatad.partial_feedback.indice_selection import generate_indices
from easydict import EasyDict
from vedatad.partial_feedback.memory_bank import load_features


chunk_size = 32
keep_ratio = 0.4
num_frames = 480
membank = EasyDict(
    chunk_size=chunk_size,
    keep_ratio=keep_ratio,
    mode="random",
    mem_bank_meta_file="data/tmp/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112/meta_val.json",
    mem_bank_dir="data/tmp/thumos14/memory_mechanism/feat_swint_15fps_128x128_crop112x112/val",
    feat_downsample=2,
)


with open(membank["mem_bank_meta_file"], "r") as f:
    membank_metas = json.load(f)

video_names = list(membank_metas.keys())

bs = 4
np.random.seed(0)
for i in range(0, len(video_names), bs):
    t1 = time.time()
    batch_video_name = video_names[i : i + bs]

    keep_indices, drop_indices = generate_indices(
        num_frames,
        membank.chunk_size,
        membank.keep_ratio,
        membank.mode,
    )

    features = []
    for video_name in batch_video_name:
        video_num_frames = membank_metas[video_name]["num_frames"]
        if video_num_frames > num_frames:
            t_shift = (
                np.random.randint(0, video_num_frames - num_frames)
                // membank["feat_downsample"]
            )
        else:
            t_shift = 0

        feature = load_features(
            mem_bank_file=os.path.join(
                membank["mem_bank_dir"],
                video_name + ".mmap",
            ),
            shape=tuple(membank_metas[video_name]["feat_shape"]),
            chunk_ids=drop_indices,
            chunk_size=membank["chunk_size"] // membank["feat_downsample"],
            f_offset=t_shift // membank["feat_downsample"],
        )
        features.append(feature)
    features = np.stack(features)
    features = to_tensor(features)
    torch.cuda.synchronize()
    t2 = time.time()
    print(drop_indices)
    print(features.shape, f"time: {t2-t1}s")
