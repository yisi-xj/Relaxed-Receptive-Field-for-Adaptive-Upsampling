_base_ = '../panoptic_fpn/panoptic-fpn_r101_fpn_1x_coco.py'

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
            guided=False)),
    test_cfg=dict(
        panoptic=dict(
            score_thr=0.6,
            max_per_img=100,
            mask_thr_binary=0.5,
            mask_overlap=0.5,
            nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True),
            stuff_area_limit=4096)))
custom_hooks = []
evaluation = dict(interval=1, metric=['PQ'])
randomness=dict(seed=0)
