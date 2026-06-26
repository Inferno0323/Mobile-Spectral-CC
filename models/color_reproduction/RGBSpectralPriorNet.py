import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SpectralPriorBottleneck(nn.Module):
    """A lightweight self-attention block that models global RGB-derived spectral priors."""

    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        norm_tokens = self.norm(tokens)
        attn_tokens, _ = self.attn(norm_tokens, norm_tokens, norm_tokens, need_weights=False)
        tokens = tokens + attn_tokens
        tokens = tokens + self.ffn(tokens)
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class RGBSpectralPriorNet(nn.Module):
    """RGB-only enhancement model trained with auxiliary multispectral supervision.

    The network consumes RGB at inference time. During training, the wrapper can
    supervise the spectral head with paired MS data so the bottleneck learns a
    spectral prior that improves RGB reconstruction/color correction.
    """

    def __init__(
        self,
        rgb_input_channels=3,
        output_channels=3,
        spectral_output_channels=15,
        base_channels=32,
        num_heads=4,
        spectral_size=64,
    ):
        super().__init__()
        self.spectral_output_channels = spectral_output_channels
        self.spectral_size = spectral_size

        self.enc0 = ConvBlock(rgb_input_channels, base_channels)
        self.enc1 = DownBlock(base_channels, base_channels * 2)
        self.enc2 = DownBlock(base_channels * 2, base_channels * 4)
        self.enc3 = DownBlock(base_channels * 4, base_channels * 8)

        bottleneck_channels = base_channels * 8
        self.prior = SpectralPriorBottleneck(bottleneck_channels, num_heads=num_heads)

        self.up2 = UpBlock(bottleneck_channels, base_channels * 4, base_channels * 4)
        self.up1 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.up0 = UpBlock(base_channels * 2, base_channels, base_channels)

        self.rgb_head = nn.Sequential(
            nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.spectral_head = nn.Sequential(
            nn.Conv2d(bottleneck_channels, bottleneck_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(bottleneck_channels // 2, spectral_output_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, rgb):
        e0 = self.enc0(rgb)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        bottleneck = self.prior(e3)

        x = self.up2(bottleneck, e2)
        x = self.up1(x, e1)
        x = self.up0(x, e0)

        enhanced_rgb = self.rgb_head(x)
        spectral = self.spectral_head(bottleneck)
        if spectral.shape[-2:] != (self.spectral_size, self.spectral_size):
            spectral = F.interpolate(spectral, size=(self.spectral_size, self.spectral_size), mode="bilinear", align_corners=False)

        return enhanced_rgb, spectral
