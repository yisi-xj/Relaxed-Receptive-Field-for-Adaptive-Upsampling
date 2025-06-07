import torch
from torch import nn
import torch.nn.functional as F


class HolisticIndexBlock(nn.Module):
    def __init__(self, inp):
        super(HolisticIndexBlock, self).__init__()

        use_nonlinear = True
        use_context = True

        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        if use_nonlinear:
            self.indexnet = nn.Sequential(
                nn.Conv2d(inp, 2 * inp, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
                nn.BatchNorm2d(2 * inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(2 * inp, 4, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            self.indexnet = nn.Conv2d(inp, 4, kernel_size=kernel_size, stride=2, padding=padding, bias=False)

    def forward(self, x):
        x = self.indexnet(x)

        y = torch.sigmoid(x)
        z = F.softmax(y, dim=1)

        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        return idx_en, idx_de


class DepthwiseO2OIndexBlock(nn.Module):
    def __init__(self, inp, use_nonlinear=False, use_context=False, batch_norm=nn.BatchNorm2d):
        super(DepthwiseO2OIndexBlock, self).__init__()

        self.indexnet1 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet2 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet3 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet4 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)

    def _build_index_block(self, inp, use_nonlinear, use_context, BatchNorm2d):

        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        if use_nonlinear:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, groups=inp, bias=False),
                BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0, groups=inp, bias=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, groups=inp, bias=False)
            )

    def forward(self, x):
        bs, c, h, w = x.size()

        x1 = self.indexnet1(x).unsqueeze(2)
        x2 = self.indexnet2(x).unsqueeze(2)
        x3 = self.indexnet3(x).unsqueeze(2)
        x4 = self.indexnet4(x).unsqueeze(2)

        x = torch.cat((x1, x2, x3, x4), dim=2)

        # normalization
        y = torch.sigmoid(x)
        z = F.softmax(y, dim=2)
        # pixel shuffling
        y = y.view(bs, c * 4, int(h / 2), int(w / 2))
        z = z.view(bs, c * 4, int(h / 2), int(w / 2))
        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        return idx_en, idx_de


class DepthwiseM2OIndexBlock(nn.Module):
    def __init__(self, inp, use_nonlinear=True, use_context=True, batch_norm=nn.BatchNorm2d):
        super(DepthwiseM2OIndexBlock, self).__init__()
        self.use_nonlinear = use_nonlinear

        self.indexnet1 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet2 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet3 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet4 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)

    def _build_index_block(self, inp, use_nonlinear, use_context, BatchNorm2d):

        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        if use_nonlinear:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
                BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                nn.Conv2d(inp, inp, kernel_size=1, stride=1, padding=0, bias=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, bias=False)
            )

    def forward(self, x):
        bs, c, h, w = x.size()

        x1 = self.indexnet1(x).unsqueeze(2)
        x2 = self.indexnet2(x).unsqueeze(2)
        x3 = self.indexnet3(x).unsqueeze(2)
        x4 = self.indexnet4(x).unsqueeze(2)

        x = torch.cat((x1, x2, x3, x4), dim=2)

        # normalization
        y = torch.sigmoid(x)
        z = F.softmax(y, dim=2)
        # pixel shuffling
        y = y.view(bs, c * 4, int(h / 2), int(w / 2))
        z = z.view(bs, c * 4, int(h / 2), int(w / 2))
        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        return idx_en, idx_de


class IndexUp(nn.Module):
    def __init__(self, in_channels, mode='hin'):
        super(IndexUp, self).__init__()
        if mode == 'hin':
            self.index_net = HolisticIndexBlock(in_channels)
        elif mode == 'o2o':
            self.index_net = DepthwiseO2OIndexBlock(in_channels)
        elif mode == 'm2o':
            self.index_net = DepthwiseM2OIndexBlock(in_channels)
        else:
            raise NotImplementedError

    def forward(self, feature_en, feature_de):
        _, mask = self.index_net(feature_en)
        mask = torch.sigmoid(mask)
        return mask * F.interpolate(feature_de, scale_factor=2)
