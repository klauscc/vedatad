import torch
from timesformer.models.vit import TimeSformer

model = TimeSformer(
    img_size=112,
    num_classes=400,
    num_frames=480,
    attention_type="divided_space_time",
    pretrained_model="",
)

model.cuda()

num_frames = 480

for i in range(10):
    dummy_video = torch.randn(
        1, 3, num_frames, 112, 112
    ).cuda()  # (batch x channels x frames x height x width)
    pred = model(dummy_video)
    loss = pred.mean()
    loss.backward()
    peak_mem_occupy = torch.cuda.max_memory_allocated() / 1024 / 1024
    peak_mem_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024

    print(
        f"itr {i}. max allocated: {peak_mem_occupy} MB, max reserved: {peak_mem_reserved}"
    )
