_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        type='UPerHead_Upsample',
        num_classes=150,
        upsample_cfg=dict(type='rrfu',
                          guided=False)
    ),
    auxiliary_head=dict(num_classes=150))

train_dataloader = dict(batch_size=8, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=4)

randomness=dict(seed=0)