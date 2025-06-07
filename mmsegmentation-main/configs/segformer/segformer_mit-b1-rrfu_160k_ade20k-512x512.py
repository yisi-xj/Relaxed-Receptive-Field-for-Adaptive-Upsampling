_base_ = ['./segformer_mit-b0_8xb2-160k_ade20k-512x512.py']

norm_cfg = dict(type='BN', requires_grad=True)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[2, 2, 2, 2]),
    decode_head=dict(
        type='SegformerHead_Upsample',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        norm_cfg=norm_cfg,
        upsample_cfg=dict(
            type='rrfu',
            guided=False
        )),  # When adding new parameters, one should modify decode_head.py
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

train_dataloader = dict(batch_size=16, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=4)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
randomness=dict(seed=0)
