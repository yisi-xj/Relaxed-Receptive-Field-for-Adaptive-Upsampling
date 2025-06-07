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
    def __init__(self, in_channels, scale_factor=2, groups=4, filter_k=3):
        super().__init__()
        self.s_f, self.s_c, self.filter_k = scale_factor, scale_factor * scale_factor, filter_k
        self.groups, self.in_channels = groups, in_channels

        self.input_proj = nn.Linear(in_channels, in_channels)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.gelu = nn.GELU()
        self.offset = nn.Linear(in_channels, groups * self.s_c * 2)
        self.mask = nn.Linear(in_channels, groups * self.s_c * 2)
        self.out_proj = nn.Linear(in_channels, in_channels)
        # self.filter_kernel = nn.Linear(in_channels, filter_k ** 2)

        self.register_buffer('coord_base', None, persistent=False)
        self.init_weights()

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.gelu(self.norm(self.input_proj(x.permute(0, 2, 3, 1).contiguous())))
        x_offset = self.offset(x_norm) * self.mask(x_norm).sigmoid()
        x_def_up_offset = x_offset.permute(0, 3, 1, 2).reshape(b, self.groups, 2, self.s_f, self.s_f, h, w).permute(
            0, 1, 2, 3, 5, 4, 6).reshape(b, self.groups * 2, self.s_f * h, self.s_f * w)
        x_up = self.out_proj(self.offset_sample(x, x_def_up_offset).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # x_up_filtering = self.filter(x_up, self.filter_kernel(x_up.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous())
        return x_up

    def init_weights(self):
        xavier_init(self.input_proj, distribution='uniform')
        normal_init(self.offset, std=0.001)
        constant_init(self.mask, val=0.0)
        # normal_init(self.filter_kernel, std=0.001)
        xavier_init(self.out_proj, distribution='uniform')

    def _init_coord_base(self, x):
        b, c, h, w = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=x.dtype) + 0.5,
            torch.arange(w, device=x.device, dtype=x.dtype) + 0.5, indexing='ij')
        self.coord_base = torch.stack([grid_x, grid_y], dim=-1).view(1, h, w, 2)

    def offset_sample(self, x, offset):
        b, c, h, w = x.shape
        b_o, c_o, h_o, w_o = offset.shape
        if self.coord_base is None or self.coord_base.shape[1:3] != offset.shape[1:3]:
            self._init_coord_base(offset)
        coords_normal = (self.coord_base + offset.view(b * self.groups, 2, h_o, w_o).permute(0, 2, 3, 1).contiguous()
                         ) / torch.tensor([w_o, h_o], device=x.device, dtype=x.dtype).view(1, 1, 1, 2)
        sample_coords = (2.0 * coords_normal - 1)
        return nn.functional.grid_sample(x.reshape(b * self.groups, -1, h, w), sample_coords, mode='bilinear',
                                         padding_mode='border', align_corners=False).view(b, -1, h_o, w_o)

    def filter(self, x, filter_kernel):
        b, c, h, w = x.shape
        filter_kernel_norm = torch.softmax(filter_kernel.view(b, self.filter_k ** 2, h, w), dim=1)
        x_unfold = nn.functional.unfold(x, self.filter_k, dilation=1, padding=self.filter_k // 2).reshape(
            b, c, self.filter_k ** 2, h, w)
        x_filtering = torch.einsum('bkhw,bckhw->bchw', [filter_kernel_norm, x_unfold]).contiguous().view(b, c, h, w)
        return x_filtering
