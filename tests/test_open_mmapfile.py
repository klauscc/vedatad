# -*- coding: utf-8 -*-
# ================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
# ================================================================

import os
import numpy as np

fps = []
data_dir = "data/tmp/test_mmapopen"
os.makedirs(data_dir, exist_ok=True)
for i in range(2000):
    filename = os.path.join(data_dir, f"test_{i}.mmap")
    fp = np.memmap(filename, dtype="int16", mode="w+", shape=(100,))
    fps.append(fp)
