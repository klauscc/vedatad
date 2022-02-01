# 1. data
dataset_type = "ANetDataset"
data_root = "data/anet/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
num_frames = 480
img_shape = (112, 112)
overlap_ratio = 0.5
img_dir = "frames_128x128_480frames"
keep_ratio = 0.4

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        typename=dataset_type,
        classes="./data/annots/anet/class_name.txt",
        ann_file="./data/annots/anet/activity_net.v1-3.min.json",
        subset="training",
        video_prefix=data_root + img_dir,
        pipeline=[
            dict(typename="LoadMetaInfo"),
            dict(typename="LoadAnnotations"),
            dict(typename="Time2Frame"),
            dict(typename="TemporalRandomCrop", num_frames=num_frames, iof_th=0.0),
            dict(typename="LoadFrames", to_float32=True, video_name_prefix="v_"),
            dict(typename="SpatialRandomCrop", crop_size=img_shape),
            dict(
                typename="PhotoMetricDistortion",
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18,
                p=0.5,
            ),
            dict(typename="Rotate", limit=(-45, 45), border_mode="reflect101", p=0.5),
            dict(typename="SpatialRandomFlip", flip_ratio=0.5),
            dict(typename="Normalize", **img_norm_cfg),
            dict(typename="Pad", size=(num_frames, *img_shape)),
            dict(typename="DefaultFormatBundle"),
            dict(
                typename="Collect",
                keys=["imgs", "gt_segments", "gt_labels", "gt_segments_ignore"],
            ),
        ],
    ),
    val=dict(
        typename=dataset_type,
        classes="./data/annots/anet/class_name.txt",
        ann_file="./data/annots/anet/activity_net.v1-3.afsd.json",
        video_prefix=data_root + img_dir,
        subset="validation",
        pipeline=[
            dict(typename="LoadMetaInfo"),
            dict(typename="Time2Frame"),
            dict(
                typename="OverlapCropAug",
                num_frames=num_frames,
                overlap_ratio=overlap_ratio,
                transforms=[
                    dict(typename="TemporalCrop"),
                    dict(
                        typename="LoadFrames", to_float32=True, video_name_prefix="v_"
                    ),
                    dict(typename="SpatialCenterCrop", crop_size=img_shape),
                    dict(typename="Normalize", **img_norm_cfg),
                    dict(typename="Pad", size=(num_frames, *img_shape)),
                    dict(typename="DefaultFormatBundle"),
                    dict(typename="Collect", keys=["imgs"]),
                ],
            ),
        ],
    ),
)

# 2. model
num_classes = 200
strides = [8, 16, 32, 64, 128, 256, 512]
use_sigmoid = True
scales_per_octave = 5
octave_base_scale = 2
num_anchors = scales_per_octave

model = dict(
    typename="SingleStageDetector",
    backbone=dict(
        typename="GradDropChunkVideoSwinV2",
        keep_ratio=keep_ratio,
        chunk_size=32,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        drop_path_rate=0.1,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        patch_norm=True,
        frozen_stages=2,
        use_checkpoint=False,
    ),
    neck=[
        dict(
            typename="SRMSwin",
            srm_cfg=dict(
                in_channels=768,
                out_channels=512,
                with_transformer=False,
            ),
        ),
        dict(
            typename="Transformer1DRelPos",
            encoder_layer_cfg=dict(
                dim=512,
                num_heads=16,
                max_seq_len=num_frames // strides[0],
                drop_path=0.1,
            ),
            num_layers=3,
        ),
        dict(
            typename="TDM",
            in_channels=512,
            stage_layers=[1] * (len(strides) - 1),
            out_channels=512,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=dict(typename="SyncBN"),
            act_cfg=dict(typename="ReLU"),
            out_indices=list(range(len(strides))),
        ),
        dict(
            typename="FPN",
            in_channels=[512] * len(strides),
            out_channels=256,
            num_outs=len(strides),
            start_level=0,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=dict(typename="SyncBN"),
        ),
    ],
    head=dict(
        typename="RetinaHead",
        num_classes=num_classes,
        num_anchors=num_anchors,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        use_sigmoid=use_sigmoid,
        conv_cfg=dict(typename="Conv1d"),
        norm_cfg=dict(typename="SyncBN"),
    ),
)

# 3. engines
meshgrid = dict(
    typename="SegmentAnchorMeshGrid",
    strides=strides,
    base_anchor=dict(
        typename="SegmentBaseAnchor",
        base_sizes=strides,
        octave_base_scale=octave_base_scale,
        scales_per_octave=scales_per_octave,
    ),
)

segment_coder = dict(
    typename="DeltaSegmentCoder", target_means=[0.0, 0.0], target_stds=[1.0, 1.0]
)

train_engine = dict(
    typename="TrainEngine",
    model=model,
    criterion=dict(
        typename="SegmentAnchorCriterion",
        num_classes=num_classes,
        meshgrid=meshgrid,
        segment_coder=segment_coder,
        reg_decoded_segment=True,
        loss_cls=dict(
            typename="FocalLoss",
            use_sigmoid=use_sigmoid,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_segment=dict(typename="DIoULoss", loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                typename="MaxIoUAssigner",
                pos_iou_thr=0.8,
                neg_iou_thr=0.5,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                ignore_wrt_candidates=True,
                iou_calculator=dict(typename="SegmentOverlaps"),
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
    ),
    optimizer=dict(
        typename="SGD",
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        paramwise_cfg=dict(custom_keys=dict(backbone={"lr_mult": 0.4})),
    ),
)

# 3.2 val engine
val_engine = dict(
    typename="ValEngine",
    model=model,
    meshgrid=meshgrid,
    converter=dict(
        typename="SegmentAnchorConverter",
        num_classes=num_classes,
        segment_coder=segment_coder,
        nms_pre=1000,
        use_sigmoid=use_sigmoid,
    ),
    num_classes=num_classes,
    test_cfg=dict(
        score_thr=0.005, nms=dict(typename="nmw", iou_thr=0.85), max_per_video=100
    ),
    use_sigmoid=use_sigmoid,
)

max_epochs = 400
lr_step = 10

# 4. hooks
hooks = [
    dict(typename="OptimizerHook"),
    dict(
        typename="CosineRestartLrSchedulerHook",
        periods=[lr_step] * (max_epochs // lr_step),
        restart_weights=[1] * (max_epochs // lr_step),
        warmup="linear",
        warmup_iters=2000,
        warmup_ratio=1e-1,
        min_lr_ratio=1e-2,
    ),
    dict(typename="EvalHook", eval_cfg=dict(mode="anet")),
    dict(typename="SnapshotHook", interval=lr_step),
    dict(typename="LoggerHook", interval=50),
]

# 5. work modes
modes = ["train"]

# 6. checkpoint
# weights = dict(filepath='open-mmlab://i3d_r50_256p_32x2x1_100e_kinetics400_rgb')
weights = dict(
    filepath="data/pretrained_models/vswin/swin_tiny_patch244_window877_kinetics400_1k_keysfrom_backbone.pth"
)
# optimizer = dict(filepath='epoch_900_optim.pth')
# meta = dict(filepath='epoch_900_meta.pth')

# 7. misc
seed = 10
dist_params = dict(backend="nccl")
log_level = "INFO"
find_unused_parameters = False
deterministic = True
