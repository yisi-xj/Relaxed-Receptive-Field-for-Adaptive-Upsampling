# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet.registry import MODELS
import torch
import torch.nn.functional as F
from ..upsamplers.defsampler import DefSampler, xavier_init
from ..upsamplers.dysample import DySample, xavier_init

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType

import mmcv
from mmengine.visualization import Visualizer

@MODELS.register_module()
class FPN_CARAFE_DEFSAMPLER(BaseModule):
    """FPN_CARAFE is a more flexible implementation of FPN. It allows more
    choice for upsample methods during the top-down pathway.

    Args:
        in_channels (list[int]): Number of channels for each input feature map.
        out_channels (int): Output channels of feature pyramids.
        num_outs (int): Number of output stages.
        start_level (int): Start level of feature pyramids.
            (Default: 0)
        end_level (int): End level of feature pyramids.
            (Default: -1 indicates the last level).
        norm_cfg (dict): Dictionary to construct and config norm layer.
        activate (str): Type of activation function in ConvModule
            (Default: None indicates w/o activation).
        order (dict): Order of components in ConvModule.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 norm_cfg=None,
                 act_cfg=None,
                 order=('conv', 'norm', 'act'),
                 mode='defsampler',
                 groups=4,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_bias = norm_cfg is None
        self.relu = nn.ReLU(inplace=False)

        self.order = order
        assert order in [('conv', 'norm', 'act'), ('act', 'conv', 'norm')]
        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.align = False
        self.defsampler_modules = nn.ModuleList()
        if self.align:
            self.defsampler_modules2 = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            if mode == 'defsampler':
                self.defsampler_modules.append(DefSampler(in_channels=self.out_channels,
                                                          scale_factor=2))
                if self.align:
                    self.defsampler_modules2.append(DefSampler(in_channels=self.out_channels,
                                                               scale_factor=None, upsample=False))
            elif mode == 'dysample':
                self.defsampler_modules.append(DySample(in_channels=self.out_channels,
                                                        scale_factor=2))

            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False,
                order=self.order)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                norm_cfg=self.norm_cfg,
                bias=self.with_bias,
                act_cfg=act_cfg,
                inplace=False,
                order=self.order)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_out_levels = (
                num_outs - self.backbone_end_level + self.start_level)
        if extra_out_levels >= 1:
            for i in range(extra_out_levels):
                in_channels = (
                    self.in_channels[self.backbone_end_level -
                                     1] if i == 0 else out_channels)
                extra_l_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    norm_cfg=norm_cfg,
                    bias=self.with_bias,
                    act_cfg=act_cfg,
                    inplace=False,
                    order=self.order)
                extra_fpn_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.with_bias,
                    act_cfg=act_cfg,
                    inplace=False,
                    order=self.order)
                self.fpn_convs.append(extra_fpn_conv)
                self.lateral_convs.append(extra_l_conv)

    def init_weights(self):
        """Initialize the weights of module."""
        super(FPN_CARAFE_DEFSAMPLER, self).init_weights()
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                xavier_init(m, distribution='uniform')
        for m in self.modules():
            if isinstance(m, DefSampler):
                m.init_weights()
            if isinstance(m, DySample):
                m.init_weights()

    def slice_as(self, src, dst):
        """Slice ``src`` as ``dst``

        Note:
            ``src`` should have the same or larger size than ``dst``.

        Args:
            src (torch.Tensor): Tensors to be sliced.
            dst (torch.Tensor): ``src`` will be sliced to have the same
                size as ``dst``.

        Returns:
            torch.Tensor: Sliced tensor.
        """
        assert (src.size(2) >= dst.size(2)) and (src.size(3) >= dst.size(3))
        if src.size(2) == dst.size(2) and src.size(3) == dst.size(3):
            return src
        else:
            return src[:, :, :dst.size(2), :dst.size(3)]

    def tensor_add(self, a, b):
        """Add tensors ``a`` and ``b`` that might have different sizes."""
        if a.size() == b.size():
            c = a + b
        else:
            c = a + self.slice_as(b, a)
        return c

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        image = mmcv.imread('.png', channel_order='rgb')
        visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')], save_dir='/newHome/S5_XJ/最新代码/mmdetection/work_dirs')

        # build laterals
        laterals = []
        for i, lateral_conv in enumerate(self.lateral_convs):
            if i <= self.backbone_end_level - self.start_level:
                input = inputs[min(i + self.start_level, len(inputs) - 1)]
            else:
                input = laterals[-1]
            lateral = lateral_conv(input)
            laterals.append(lateral)

            drawn_img = visualizer.draw_featmap(lateral, image=image, channel_reduction='select_max')
            visualizer.add_image('特征图可视化原图', drawn_img, step=i)

        # build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            upsample_feat = self.defsampler_modules[i-1](laterals[i])
            if self.align:
                laterals[i - 1] = self.defsampler_modules2[i-1](laterals[i - 1])
            laterals[i - 1] = self.tensor_add(laterals[i - 1], upsample_feat)

            drawn_img = visualizer.draw_featmap(upsample_feat, image=image, channel_reduction='select_max')
            visualizer.add_image('特征图可视化上采样', drawn_img, step=i)

            drawn_img = visualizer.draw_featmap(laterals[i - 1], image=image, channel_reduction='select_max')
            visualizer.add_image('特征图可视化融合', drawn_img, step=i)

        # build outputs
        num_conv_outs = len(self.fpn_convs)
        outs = []
        for i in range(num_conv_outs):
            out = self.fpn_convs[i](laterals[i])
            outs.append(out)

            drawn_img = visualizer.draw_featmap(out, image=image, channel_reduction='select_max')
            visualizer.add_image('特征图可视化融合后卷积', drawn_img, step=i)
            
        drawn_img = visualizer.draw_featmap(outs, image=image, channel_reduction='select_max')
        visualizer.add_image('特征图可视化结果', drawn_img, step=1)

        return tuple(outs)