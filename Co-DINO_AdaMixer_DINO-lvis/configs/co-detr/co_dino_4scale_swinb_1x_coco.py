_base_ = './co_dino_4scale_r50_1x_coco.py'
pretrained = '../pretrained_models/swin_base_patch4_window7_224_22k.pth'
model = dict(
    change_foundation_model_size=448,
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV1',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        pretrained=pretrained,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    neck=dict(in_channels=[256, 512, 1024])
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-4,
    weight_decay=0.05,
    # custom_keys of sampling_offsets and reference_points in DeformDETR
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                    # Swin-related settings
                                    'absolute_pos_embed': dict(decay_mult=0.),
                                    'relative_position_bias_table': dict(decay_mult=0.),
                                    'norm': dict(decay_mult=0.)}))


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 1200), (512, 1200), (544, 1200),
                               (576, 1200), (608, 1200), (640, 1200),
                               (672, 1200), (704, 1200), (736, 1200),
                               (768, 1200), (800, 1200)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='Resize',
                    img_scale=[(480, 1200), (512, 1200), (544, 1200),
                               (576, 1200), (608, 1200), (640, 1200),
                               (672, 1200), (704, 1200), (736, 1200),
                               (768, 1200), (800, 1200)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]


dataset_type = 'CocoDataset'
data_root = '../dataset/coco/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        filter_empty_gt=False, 
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
