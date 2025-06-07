import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from depth.models.builder import HEADS
from .decode_head import DepthBaseDecodeHead
import torch.nn.functional as F
from ..upsamplers import build_upsampler


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

    def forward(self, x, concat_with):
        if self.guided:
            up_x = self.upsampler(concat_with, x)
        else:
            up_x = self.upsampler(x)
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
                    temp_feat = self.conv_list[index](feat)
                    temp_feat_list.append(temp_feat)
                else:
                    skip_feat = feat
                    up_feat = temp_feat_list[index-1]
                    temp_feat = self.conv_list[index](up_feat, skip_feat)
                    temp_feat_list.append(temp_feat)

        output = self.depth_pred(temp_feat_list[-1])
        return output
