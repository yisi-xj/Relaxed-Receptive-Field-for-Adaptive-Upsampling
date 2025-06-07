import torch
import torch.nn as nn


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


class DefSampler(nn.Module):
    def __init__(self, in_channels, scale_factor=2, groups=4, filter_k=3, guided=False):
        super().__init__()
        self.s_f, self.s_c, self.filter_k = scale_factor, scale_factor * scale_factor, filter_k
        self.groups, self.guided = groups, guided

        self.def_offset_conv = nn.Conv2d(in_channels, groups * self.s_c * 2, kernel_size=1, padding=1 // 2)
        self.def_assistant_conv = nn.Conv2d(in_channels, groups * self.s_c * 2, kernel_size=1, padding=1 // 2)
        if self.guided:
            self.compress_conv = nn.Conv2d(in_channels * 2, 64, kernel_size=1, padding=1 // 2)
            # self.compress_conv = nn.Conv2d(in_channels, 64, kernel_size=1, padding=1 // 2)
            self.h_filter_kernel_conv = nn.Conv2d(64, filter_k ** 2, kernel_size=1, padding=1 // 2)
        else:
            self.compress_conv = nn.Conv2d(in_channels, 64, kernel_size=1, padding=1 // 2)
        self.filter_kernel_conv = nn.Conv2d(64, filter_k ** 2, kernel_size=1, padding=1 // 2)

        self.trim_conv = nn.Conv2d(64, groups * 2, kernel_size=1, padding=1 // 2)
        self.trim_assistant_conv = nn.Conv2d(64, groups * 2, kernel_size=1, padding=1 // 2)

        self.register_buffer('def_coord_base', None, persistent=False)
        self.register_buffer('trim_coord_base', None, persistent=False)
        self.init_weights()

    def forward(self, x, x_h=None):
        x_def_offset = self.def_offset_conv(x) * self.def_assistant_conv(x).sigmoid()
        x_def_up = self.def_sample(x, x_def_offset)
        if self.guided:
            # compress_x_up = self.compress_conv(x_def_up + x_h)
            compress_x_up = self.compress_conv(torch.cat([x_def_up, x_h], dim=1))
        else:
            compress_x_up = self.compress_conv(x_def_up)
        x_up_filtering = self.filter(x_def_up, compress_x_up, self.filter_kernel_conv)
        if self.guided:
            x_h_filtering = self.filter(x_h, compress_x_up, self.h_filter_kernel_conv)
            x_up_filtering = x_up_filtering + (x_h - x_h_filtering)
        x_def_up_trim = self.trim(x_up_filtering, self.trim_conv(compress_x_up) * self.trim_assistant_conv(compress_x_up).sigmoid())
        return x_def_up_trim

    def init_weights(self):
        normal_init(self.def_offset_conv, std=0.001)
        constant_init(self.def_assistant_conv, val=0.0)
        xavier_init(self.compress_conv, distribution='uniform')
        normal_init(self.filter_kernel_conv, std=0.001)
        if self.guided:
            normal_init(self.h_filter_kernel_conv, std=0.001)
        normal_init(self.trim_conv, std=0.001)
        constant_init(self.trim_assistant_conv, val=0.0)

    def _init_def_coord_base(self, x_def_offset):
        b_d_g, h_d, w_d, yx = x_def_offset.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h_d, device=x_def_offset.device, dtype=x_def_offset.dtype) + 0.5,
            torch.arange(w_d, device=x_def_offset.device, dtype=x_def_offset.dtype) + 0.5, indexing='ij')
        self.def_coord_base = torch.stack([grid_x, grid_y], dim=-1).view(1, h_d, w_d, 2)

    def _init_trim_coord_base(self, x):
        b, c, h, w = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype) + 0.5,
            torch.arange(w, device=x.device, dtype=x.dtype) + 0.5, indexing='ij')
        self.trim_coord_base = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, h, w, 2)

    def filter(self, x, compress, filter_kernel_conv):
        b, c, h, w = x.shape
        filter_kernel_norm = torch.softmax(
            filter_kernel_conv(compress).view(b, self.filter_k ** 2, h, w), dim=1)
        x_unfold = nn.functional.unfold(x, self.filter_k, dilation=1, padding=self.filter_k // 2).view(
            b, c, self.filter_k ** 2, h, w)
        x_filtering = torch.einsum('bkhw,bckhw->bchw', [filter_kernel_norm, x_unfold]).contiguous().view(b, c, h, w)
        return x_filtering
    def def_sample(self, x, x_def_offset):
        b, c, h, w = x.shape
        x_def_offset = x_def_offset.reshape(b * self.groups, self.s_f, self.s_f, 2, h, w).permute(
            0, 4, 1, 5, 2, 3).contiguous().view(b * self.groups, self.s_f * h, self.s_f * w, 2)
        b_d_g, h_d, w_d, yx = x_def_offset.shape
        if self.def_coord_base is None or self.def_coord_base.shape[1:3] != x_def_offset.shape[1:3]:
            self._init_def_coord_base(x_def_offset)
        coords_normal = (self.def_coord_base + x_def_offset) / torch.tensor(
            [w_d, h_d], device=x.device, dtype=x.dtype).view(1, 1, 1, 2)
        def_coords = (2.0 * coords_normal - 1)
        return nn.functional.grid_sample(x.reshape(b * self.groups, -1, h, w), def_coords, mode='bilinear',
                                         padding_mode='border', align_corners=False).view(b, -1, h_d, w_d)

    def trim(self, x, trim_offset):
        b, c, h, w = x.shape
        if self.trim_coord_base is None or self.trim_coord_base.shape[2:4] != trim_offset.shape[2:4]:
            self._init_trim_coord_base(trim_offset)
        coords_normal = (self.trim_coord_base + trim_offset.view(b, self.groups, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
                         ) / torch.tensor([w, h], device=x.device, dtype=x.dtype).view(1, 1, 1, 1, 2)
        trim_coords = (2.0 * coords_normal - 1).view(b * self.groups, h, w, 2)
        return nn.functional.grid_sample(x.reshape(b * self.groups, -1, h, w), trim_coords, mode='bilinear',
                                         padding_mode='border', align_corners=False).view(b, -1, h, w)