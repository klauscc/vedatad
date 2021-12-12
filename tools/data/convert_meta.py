# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================

import json


meta_file = "data/thumos14/memory_mechanism/feat_swinb_15fps_256x256_crop224x224/meta_val.json"

meta = json.load(open(meta_file, "r"))
new_meta = {}
for video in meta:
    key = list(video.keys())[0]
    new_meta[key] = video[key]
with open(meta_file, "w") as f:
    json.dump(new_meta, f, indent=4)
