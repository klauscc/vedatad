import json
import os
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from einops.einops import rearrange
from PIL import Image
from timesformer.models.vit import TimeSformer

config = dict(tpename="GradDropTimeSformer")

model = TimeSformer(
    img_size=224,
    num_classes=600,
    num_frames=96,
    attention_type="divided_space_time",
    pretrained_model="",
)

pretrained_model = (
    "data/pretrained_models/timesformer/TimeSformer_divST_96x4_224_K600.pyth"
)
states = torch.load(pretrained_model)
info = model.load_state_dict(states["model_state"])
model.cuda()
model.eval()

device = torch.device("cuda:0")

split = "val"
video_dir = f"data/thumos14/frames_15fps_256x256/{split}"
dst_dir = (
    f"data/thumos14/memory_mechanism/timesformer_96x4_15fps_256x256_crop224x224/{split}"
)
meta_file = f"data/thumos14/memory_mechanism/timesformer_96x4_15fps_256x256_crop224x224/meta_{split}.json"
BATCH_SIZE = 8
CHUNK_SIZE = 96
IMG_SHAPE = (224, 224)
FEAT_DIM = 768
IMG_MEAN = torch.tensor([114.75, 114.75, 114.75], device=device)
IMG_STD = torch.tensor([57.375, 57.375, 57.375], device=device)
########################################

os.makedirs(dst_dir, exist_ok=True)


def load_video(video_path):
    """load frames

    Args:
        video_path (string): path contains the frames.

    Returns: list of image. Each image is of shape (H,W,3).

    """
    frame_paths = sorted(glob(os.path.join(video_path, "*.png")))
    imgs = []
    for img_path in frame_paths:
        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img).astype(np.float32)
        imgs.append(img)
    return imgs


def extract_one_video(video_path, dst_path):
    """extract features for one video

    Args:
        video_path (string): path contains the frames.
        dst_path (string): the memmap file to save the features.

    Returns: tuple[T,C]. the feature shape for video.

    """
    imgs = load_video(video_path)
    num_frames = len(imgs)
    input_length = BATCH_SIZE * CHUNK_SIZE
    features = []
    for frame_idx in range(0, num_frames, input_length):
        frames = imgs[frame_idx : frame_idx + input_length]
        frames = np.stack(frames, axis=0)  # [T,H,W,C]
        # to torch.Tensor
        frames = torch.from_numpy(frames).cuda()

        # pad
        pad_r = 0
        if len(frames) < input_length:
            pad_r = input_length - len(frames)
            frames = F.pad(frames, (0, 0, 0, 0, 0, 0, 0, pad_r))

        frames = frames.permute(3, 0, 1, 2)  # [C,T,H,W]
        frames = tf.center_crop(frames, IMG_SHAPE)
        frames = (frames - IMG_MEAN.view(3, 1, 1, 1)) / IMG_STD.view(3, 1, 1, 1)
        frames = rearrange(frames, "C (B L) H W -> B C L H W", L=CHUNK_SIZE)
        with torch.no_grad():
            feat = model.model.forward_features(frames, temporal_pooling=False)
            feat = feat[:, 1:]
            feat = rearrange(feat, "b (n t) m -> b n t m", t=CHUNK_SIZE)
            feat = feat.mean(1)  # [b, t, m]
            feat = rearrange(feat, "b t m -> (b t) m")
            if pad_r != 0:
                feat = feat[: input_length - pad_r]
        features.append(feat.cpu().numpy())

    features = np.concatenate(features)
    fp = np.memmap(dst_path, dtype="float32", mode="w+", shape=features.shape)
    fp[:] = features[:]
    fp.flush()
    return num_frames, features.shape


video_paths = sorted(glob(os.path.join(video_dir, "*")))
metas = {}
for i, p in enumerate(video_paths):
    video_name = os.path.basename(p)
    dst_path = os.path.join(dst_dir, video_name + ".mmap")
    print(f"{i}/{len(video_paths)}: extract features for video: {video_name}")
    num_frames, feat_shape = extract_one_video(p, dst_path)
    print(f"    num_frames:{num_frames}. feat_shape: {feat_shape}")
    metas[video_name] = {"num_frames": num_frames, "feat_shape": feat_shape}

with open(meta_file, "w") as f:
    json.dump(metas, f, indent=4)
