import torch.nn as nn


class PixelUpsampler(nn.Module):
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.sampler_num = self.scale_factor ** 2
        self.sampler_conv = nn.Conv2d(in_channels, self.sampler_num, kernel_size=3, stride=1, padding=1)
        # nn.init.normal_(self.sampler_conv.weight, mean=0, std=0.005)
        nn.init.xavier_uniform_(self.sampler_conv.weight, nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(self.sampler_conv.bias, 0.0)
        self.leaky_relu = nn.LeakyReLU()
        self.pixel_shuffle = nn.PixelShuffle(self.scale_factor)

    def forward(self, x):
        sampler = self.leaky_relu(self.sampler_conv(x))
        b, s_c, h, w = sampler.shape
        # (b 1 4 h w) * (b c 1 h w) -> (b c 4 h w)
        sample_channels = sampler.view(b, 1, self.sampler_num, h, w) * x.view(b, -1, 1, h, w)
        x = self.pixel_shuffle(sample_channels.view(b, -1, h, w))
        return x

