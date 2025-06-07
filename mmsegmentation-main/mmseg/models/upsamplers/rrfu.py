import torch
import torch.nn as nn
import torch.nn.functional as F


def xavier_init(module, gain=1.0, bias=0.0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0.0, std=1.0, bias=0.0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0.0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class RRFU(nn.Module):
    def __init__(self, in_channels, scale_factor=2, groups=4, sample_k=3):
        super().__init__()
        self.s_f, self.s_c = scale_factor, scale_factor * scale_factor
        self.sample_k, self.sample_c = sample_k, sample_k * sample_k
        self.groups = groups

        self.offset = nn.Conv2d(in_channels, groups * self.s_c * 2, kernel_size=1)
        self.offset_mask = nn.Conv2d(in_channels, groups * self.s_c * 2, kernel_size=1)
        self.sample_kernel = nn.Conv2d(in_channels, self.sample_c, kernel_size=1)
        self.kernel_mask = nn.Conv2d(in_channels, self.sample_c, kernel_size=1)

        self.register_buffer('coord_base', None, persistent=False)
        self.init_weights()

    def forward(self, x):
        b, c, h, w = x.shape
        x_offset = self.offset(x) * self.offset_mask(x).sigmoid()
        x_def_offset = x_offset.reshape(b * self.groups, self.s_f, self.s_f, 2, h, w).permute(
            0, 4, 1, 5, 2, 3).reshape(b * self.groups, self.s_f * h, self.s_f * w, 2)
        x_def = self.offset_sample(x, x_def_offset)
        x_up = self.dynamic_sample(x_def, self.sample_kernel(x_def) * self.kernel_mask(x_def).sigmoid())
        return x_up

    def init_weights(self):
        normal_init(self.offset, std=0.001)
        constant_init(self.offset_mask, val=0.0)
        normal_init(self.sample_kernel, std=0.005)
        constant_init(self.kernel_mask, val=0.0)

    def _init_coord_base(self, x):
        bg, h, w, xy = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype) + 0.5,
            torch.arange(w, device=x.device, dtype=x.dtype) + 0.5, indexing='ij')
        self.coord_base = torch.stack([grid_x, grid_y], dim=-1).view(1, h, w, 2)

    def offset_sample(self, x, offset):
        b, c, h, w = x.shape
        bg_o, h_o, w_o, xy = offset.shape
        if self.coord_base is None or self.coord_base.shape[1:3] != offset.shape[1:3]:
            self._init_coord_base(offset)
        grid = 2.0 * ((self.coord_base + offset) / torch.tensor([w_o, h_o], device=x.device, dtype=x.dtype).view(
            1, 1, 1, 2)) - 1
        x_def_sample = F.grid_sample(x.reshape(b * self.groups, -1, h, w), grid, mode='bilinear',
                                     padding_mode='border', align_corners=False).view(b, c, h_o, w_o)
        return x_def_sample

    def dynamic_sample(self, x, sample_kernel):
        b, c, h, w = x.shape
        sample_kernel_norm = torch.softmax(sample_kernel, dim=1)
        x_unfold = F.unfold(x, self.sample_k, dilation=1, padding=self.sample_k // 2).reshape(b, c, self.sample_c, h, w)
        x_sample = torch.einsum('bkhw,bckhw->bchw', [sample_kernel_norm, x_unfold]).contiguous()
        return x_sample