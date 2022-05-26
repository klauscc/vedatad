import torch
from timesformer.models.vit import TimeSformer

num_frames = 96

model = TimeSformer(
    img_size=224,
    num_classes=600,
    num_frames=num_frames,
    attention_type="divided_space_time",
    pretrained_model="",
)

model.cuda()


for i in range(10):
    dummy_video = torch.randn(
        1, 3, num_frames, 224, 224
    ).cuda()  # (batch x channels x frames x height x width)
    pred = model(dummy_video)
    loss = pred.mean()
    loss.backward()
    peak_mem_occupy = torch.cuda.max_memory_allocated() / 1024 / 1024
    peak_mem_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024

    print(
        f"itr {i}. max allocated: {peak_mem_occupy} MB, max reserved: {peak_mem_reserved}"
    )
