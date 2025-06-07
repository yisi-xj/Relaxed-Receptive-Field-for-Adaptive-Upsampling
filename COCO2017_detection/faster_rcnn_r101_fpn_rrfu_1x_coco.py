_base_ = '../faster_rcnn/faster-rcnn_r101_fpn_1x_coco.py'
model = dict(
    data_preprocessor=dict(pad_size_divisor=64),
    neck=dict(
        type='FPN_CARAFE_Upsample',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        start_level=0,
        end_level=-1,
        norm_cfg=None,
        act_cfg=None,
        order=('conv', 'norm', 'act'),
        upsample_cfg=dict(
            type='rrfu',
            guided=False)))

randomness=dict(seed=0)
