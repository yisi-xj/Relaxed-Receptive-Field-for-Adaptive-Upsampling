# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from ..utils import resize
from ..upsamplers import build_upsampler

import os
import os.path as osp
import numpy as np
from PIL import Image
class FeatureVisualizer:
    """特征图可视化处理器（适配旧版目录结构）"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.lock = torch.multiprocessing.Lock()

    def _mkdir(self, pth):
        if not os.path.exists(pth):
            os.makedirs(pth)

    def _normalize(self, img):
        return (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0

    def save_features(self, features: torch.Tensor, index:int, stage: str):
        if features.dim() != 4:
            return

        with self.lock:
            # 创建以样本编号为名的子目录
            output_dir_index = osp.join(self.output_dir, str(index))
            sum_dir = osp.join(output_dir_index, 'sum')
            std_dir = osp.join(output_dir_index, 'std')
            avg_dir = osp.join(output_dir_index, 'avg')
            max_dir = osp.join(output_dir_index, 'max')
            self._mkdir(sum_dir)
            self._mkdir(std_dir)
            self._mkdir(avg_dir)
            self._mkdir(max_dir)

            # 处理特征图
            feat_np = features.squeeze(0).detach().cpu().numpy()
            sum_map = self._normalize(feat_np.sum(axis=0))
            std_map = self._normalize(feat_np.std(axis=0))
            avg_map = self._normalize(feat_np.mean(axis=0))
            max_map = self._normalize(feat_np.max(axis=0))

            # 按阶段保存（stage_0.png, stage_1.png...）
            Image.fromarray(np.uint8(sum_map)).save(osp.join(sum_dir, f'stage_{stage}.png'))
            Image.fromarray(np.uint8(std_map)).save(osp.join(std_dir, f'stage_{stage}.png'))
            Image.fromarray(np.uint8(avg_map)).save(osp.join(avg_dir, f'stage_{stage}.png'))
            Image.fromarray(np.uint8(max_map)).save(osp.join(max_dir, f'stage_{stage}.png'))


@MODELS.register_module()
class SegformerHead_Upsample(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self,
                 upsample_cfg=dict(
                     mode='bilinear',
                     guided=False,
                     align_corners=False
                 ),
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.record = 0

        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.upsample_cfg = upsample_cfg.copy()
        self.guided_upsample = self.upsample_cfg['guided']
        self.upsample_mode = self.upsample_cfg['type']
        self.upsample_modules = nn.ModuleList()
        if self.upsample_mode == 'bilinear' or self.upsample_mode == 'nearest':
            self.upsample_stages = 3
        else:
            self.upsample_stages = 6
        for i in range(self.upsample_stages):
            self.upsample_modules.append(build_upsampler(self.upsample_cfg, in_channels=self.channels, scale_factor=2))

    def forward(self, inputs):
        self.record += 1
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        featureVisualizer = FeatureVisualizer('/newHome/S6_XJ/mmsegmentation-main/work_dirs/vis_show')
        inputs = self._transform_inputs(inputs)
        x0, x1, x2, x3 = inputs

        featureVisualizer.save_features(x0, self.record, 'x0')
        featureVisualizer.save_features(x1, self.record, 'x1')
        featureVisualizer.save_features(x2, self.record, 'x2')
        featureVisualizer.save_features(x3, self.record, 'x3')

        x0 = self.convs[0](x0)
        x1 = self.convs[1](x1)
        x2 = self.convs[2](x2)
        x3 = self.convs[3](x3)

        featureVisualizer.save_features(x0, self.record, 'x0_conv')
        featureVisualizer.save_features(x1, self.record, 'x1_conv')
        featureVisualizer.save_features(x2, self.record, 'x2_conv')
        featureVisualizer.save_features(x3, self.record, 'x3_conv')

        if x1.shape[2] * 2 != x0.shape[2]  or x1.shape[3] * 2 != x0.shape[3]:
            print('不一致x0, x1, x2, x3形状', x0.shape, x1.shape, x2.shape, x3.shape)

        if self.upsample_mode == 'bilinear':
            x3 = resize(x3, size=x0.size()[2:], mode='bilinear', align_corners=False)
            x2 = resize(x2, size=x0.size()[2:], mode='bilinear', align_corners=False)
            x1 = resize(x1, size=x0.size()[2:], mode='bilinear', align_corners=False)
        elif self.upsample_mode == 'nearest':
            x3 = resize(x3, size=x0.size()[2:], mode='nearest')
            x2 = resize(x2, size=x0.size()[2:], mode='nearest')
            x1 = resize(x1, size=x0.size()[2:], mode='nearest')
        elif self.guided_upsample:
            x3 = self.upsample_modules[0](x2, x3)
            x3 = self.upsample_modules[1](x1, x3)
            x3 = self.upsample_modules[2](x0, x3)
            x2 = self.upsample_modules[3](x1, x2)
            x2 = self.upsample_modules[4](x0, x2)
            x1 = self.upsample_modules[5](x0, x1)
        elif self.upsample_mode == 'rrfu':
            x3 = self.upsample_modules[0](x3, featureVisualizer, self.record, 'x3_up1')
            x3 = self.upsample_modules[1](x3, featureVisualizer, self.record, 'x3_up2')
            x3 = self.upsample_modules[2](x3, featureVisualizer, self.record, 'x3_up3')
            x2 = self.upsample_modules[3](x2, featureVisualizer, self.record, 'x2_up1')
            x2 = self.upsample_modules[4](x2, featureVisualizer, self.record, 'x2_up2')
            x1 = self.upsample_modules[5](x1, featureVisualizer, self.record, 'x1_up1')
        else:
            x3 = self.upsample_modules[0](x3)
            x3 = self.upsample_modules[1](x3)
            x3 = self.upsample_modules[2](x3)
            x2 = self.upsample_modules[3](x2)
            x2 = self.upsample_modules[4](x2)
            x1 = self.upsample_modules[5](x1)

        featureVisualizer.save_features(x1, self.record, 'x1_up')
        featureVisualizer.save_features(x2, self.record, 'x2_up')
        featureVisualizer.save_features(x3, self.record, 'x3_up')

        out = self.fusion_conv(torch.cat([x0, x1, x2, x3], dim=1))

        out = self.cls_seg(out)

        featureVisualizer.save_features(out, self.record, 'out')

        return out