import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from depth.models.builder import HEADS
from .decode_head import DepthBaseDecodeHead
import torch.nn.functional as F
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


class UpSample(nn.Module):
    '''Fusion module

    From Adabins

    '''

    def __init__(self,
                 in_channel,
                 skip_channel,
                 output_features,
                 upsample_cfg=dict(
                     type='bilinear',
                     guided=False,
                     align_corners=True
                 ),
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(UpSample, self).__init__()
        self.upsample_cfg = upsample_cfg.copy()
        self.guided = upsample_cfg['guided']
        self.convA = ConvModule(in_channel + skip_channel, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.upsampler = build_upsampler(self.upsample_cfg, in_channels=in_channel, scale_factor=2, y_channel=skip_channel)

    def forward(self, x, concat_with, featureVisualizer, index, record=None):
        if self.guided:
            featureVisualizer.save_features(x, record, 'x'+str(index)+'原')
            up_x = self.upsampler(concat_with, x)
            featureVisualizer.save_features(up_x, record, 'x'+str(index)+'up')
        else:
            featureVisualizer.save_features(x, record, 'x'+str(index)+'原')
            up_x = self.upsampler(x)
            featureVisualizer.save_features(up_x, record, 'x'+str(index)+'up')
        return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))


@HEADS.register_module()
class DenseDepthHead_Upsample(DepthBaseDecodeHead):
    """DenseDepthHead.
    This head is implemented of `DenseDepth: <https://arxiv.org/abs/1812.11941>`_.
    Args:
        up_sample_channels (List): Out channels of decoder layers.
        fpn (bool): Whether apply FPN head.
            Default: False
        conv_dim (int): Default channel of features in FPN head.
            Default: 256.
    """

    def __init__(self,
                 up_sample_channels,
                 fpn=False,
                 conv_dim=256,
                 upsample_cfg=dict(
                     type='bilinear',
                     guided=False,
                     align_corners=True
                 ),
                 **kwargs):
        super(DenseDepthHead_Upsample, self).__init__(**kwargs)

        self.record = 0
        self.up_sample_channels = up_sample_channels[::-1]
        self.in_channels = self.in_channels[::-1]
        self.upsample_cfg = upsample_cfg

        self.conv_list = nn.ModuleList()
        up_channel_temp = 0

        self.fpn = fpn
        if self.fpn:
            self.num_fpn_levels = len(self.in_channels)

            # construct the FPN
            self.lateral_convs = nn.ModuleList()
            self.output_convs = nn.ModuleList()

            for idx, in_channel in enumerate(self.in_channels[:self.num_fpn_levels]):
                lateral_conv = ConvModule(
                    in_channel, conv_dim, kernel_size=1, norm_cfg=self.norm_cfg
                )
                output_conv = ConvModule(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
                self.lateral_convs.append(lateral_conv)
                self.output_convs.append(output_conv)

        else:
            for index, (in_channel, up_channel) in enumerate(
                    zip(self.in_channels, self.up_sample_channels)):
                if index == 0:
                    print('in_channel, up_channel:', in_channel, up_channel)
                    self.conv_list.append(
                        ConvModule(
                            in_channels=in_channel,
                            out_channels=up_channel,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            act_cfg=None
                        ))
                else:
                    self.conv_list.append(
                        UpSample(in_channel=up_channel_temp,
                                 skip_channel=in_channel,
                                 output_features=up_channel,
                                 upsample_cfg=self.upsample_cfg,
                                 norm_cfg=self.norm_cfg,
                                 act_cfg=self.act_cfg))

                # save earlier fusion target
                up_channel_temp = up_channel

    def forward(self, inputs, img_metas):
        """Forward function."""

        self.record += 1
        featureVisualizer = FeatureVisualizer('/newHome/S6_XJ/Monocular-Depth-Estimation-Toolbox/work_dirs/show_vis2')
        temp_feat_list = []
        if self.fpn:

            for index, feat in enumerate(inputs[::-1]):
                x = feat
                lateral_conv = self.lateral_convs[index]
                output_conv = self.output_convs[index]
                cur_fpn = lateral_conv(x)

                # Following FPN implementation, we use nearest upsampling here. Change align corners to True.
                if index != 0:
                    y = cur_fpn + F.interpolate(temp_feat_list[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=True)
                else:
                    y = cur_fpn

                y = output_conv(y)
                temp_feat_list.append(y)

        else:
            temp_feat_list = []
            for index, feat in enumerate(inputs[::-1]):
                if index == 0:
                    featureVisualizer.save_features(feat, self.record, 'x'+str(index)+'原')
                    temp_feat = self.conv_list[index](feat)
                    featureVisualizer.save_features(temp_feat, self.record, 'x'+str(index)+'conv')
                    temp_feat_list.append(temp_feat)
                else:
                    skip_feat = feat
                    up_feat = temp_feat_list[index-1]
                    temp_feat = self.conv_list[index](up_feat, skip_feat, featureVisualizer, index, self.record)
                    temp_feat_list.append(temp_feat)

        output = self.depth_pred(temp_feat_list[-1])
        return output
