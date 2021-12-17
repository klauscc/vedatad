#!/usr/bin/env python

import torch
from timm import create_model


def setup_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup_random_seed(0)

device = torch.device("cpu:0")
model = create_model("swin_tiny_patch4_window7_224").to(device)

x = torch.rand([8, 3, 224, 224]).to(device)


cpu_rng = torch.get_rng_state()
gpu_rng = torch.cuda.get_rng_state()

model.train()
with torch.no_grad():
    y = model(x)
with torch.no_grad():
    y1 = model(x)


print("first:", y[1, :10])
print("second:", y1[1, :10])
print("diff", (y - y1).abs().mean())

print("------restore rng state------------")

with torch.no_grad():
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(gpu_rng)
    y = model(x)
    print(y.shape)
with torch.no_grad():
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(gpu_rng)
    y1 = model(x[1:2].contiguous())



print("first:", y[1, :10])

print("second:", y1[0, :10])
