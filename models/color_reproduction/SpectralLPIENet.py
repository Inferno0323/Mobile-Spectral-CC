import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List



class AttentionBlock(nn.Module):
    def __init__(self, dim: int):
        super(AttentionBlock, self).__init__()
        self._spatial_attention_conv = nn.Conv2d(2, dim, kernel_size=3, padding=1)

        # Channel attention MLP
        self._channel_attention_conv0 = nn.Conv2d(1, dim, kernel_size=1, padding=0)
        self._channel_attention_conv1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)

        self._out_conv = nn.Conv2d(2 * dim, dim, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):
        if len(x.shape) != 4:
            raise ValueError(f"Expected [B, C, H, W] input, got {x.shape}.")

        # Spatial attention
        mean = torch.mean(x, dim=1, keepdim=True)  # Mean/Max on C axis
        max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.cat([mean, max], dim=1)  # [B, 2, H, W]
        spatial_attention = self._spatial_attention_conv(spatial_attention)
        spatial_attention = torch.sigmoid(spatial_attention) * x

        # Channel attention. TODO: Correct that it only uses average pool contrary to CBAM?
        # NOTE/TODO: This differs from CBAM as it uses Channel pooling, not spatial pooling!
        # In a way, this is 2x spatial attention
        channel_attention = torch.relu(self._channel_attention_conv0(mean))
        channel_attention = self._channel_attention_conv1(channel_attention)
        channel_attention = torch.sigmoid(channel_attention) * x

        attention = torch.cat([spatial_attention, channel_attention], dim=1)  # [B, 2*dim, H, W]
        attention = self._out_conv(attention)
        return x + attention


# TODO: This is not named in the paper right?
# It is sort of the InverseResidualBlock but w/o the Channel and Spatial Attentions and without another Conv after ReLU
class InverseBlock(nn.Module):
    def __init__(self, input_channels: int, channels: int):
        super(InverseBlock, self).__init__()

        self._conv0 = nn.Conv2d(input_channels, channels, kernel_size=1)
        self._dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self._conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self._conv2 = nn.Conv2d(input_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        features = self._conv0(x)
        features = F.elu(self._dw_conv(features))  # TODO: Paper is ReLU, authors do ELU
        features = self._conv1(features)

        # TODO: The BaseBlock has residuals and one path of convolutions, not 2 separate paths - is this different on purpose?
        x = torch.relu(self._conv2(x))
        return x + features


class BaseBlock(nn.Module):
    def __init__(self, channels: int):
        super(BaseBlock, self).__init__()

        self._conv0 = nn.Conv2d(channels, channels, kernel_size=1)
        self._dw_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self._conv1 = nn.Conv2d(channels, channels, kernel_size=1)

        self._conv2 = nn.Conv2d(channels, channels, kernel_size=1)
        self._conv3 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        features = self._conv0(x)
        features = F.elu(self._dw_conv(features))  # TODO: ELU or ReLU?
        features = self._conv1(features)
        x = x + features

        features = F.elu(self._conv2(x))
        features = self._conv3(features)
        return x + features


class AttentionTail(nn.Module):
    def __init__(self, channels: int):
        super(AttentionTail, self).__init__()

        self._conv0 = nn.Conv2d(channels, channels, kernel_size=7, padding=3)
        self._conv1 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self._conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        attention = torch.relu(self._conv0(x))
        attention = torch.relu(self._conv1(attention))
        attention = torch.sigmoid(self._conv2(attention))
        return x * attention

class IlluminantHead(nn.Module):
    def __init__(self, in_ch: int, hidden: int):
        super(IlluminantHead, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling

        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class SpectralLPIENet(nn.Module):
    def __init__(self, rgb_input_channels: int, spectral_input_channels: int, output_channels: int, encoder_dims: List[int], decoder_dims: List[int], illuminant_head: bool = False, illuminant_hidden: int = 128, finetune: bool = False):
        super(SpectralLPIENet, self).__init__()

        if len(encoder_dims) != len(decoder_dims) + 1 or len(decoder_dims) < 1:
            raise ValueError(f"Unexpected encoder and decoder dims: {encoder_dims}, {decoder_dims}.")

        if rgb_input_channels != output_channels:
            raise NotImplementedError()

        # TODO: We will need an explicit decoder head, consider Unshuffle & Shuffle

        rgb_encoders = []
        for i, encoder_dim in enumerate(encoder_dims):
            input_dim = rgb_input_channels if i == 0 else encoder_dims[i - 1]
            rgb_encoders.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, encoder_dim, kernel_size=3, padding=1),
                    BaseBlock(encoder_dim),  # TODO: one or two base blocks?
                    BaseBlock(encoder_dim),
                    AttentionBlock(encoder_dim),
                )
            )
        self._rgb_encoders = nn.ModuleList(rgb_encoders)

        # Spectral Encoders
        spectral_encoders = []
        for i, encoder_dim in enumerate(encoder_dims):
            input_dim = spectral_input_channels if i == 0 else encoder_dims[i - 1]
            spectral_encoders.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, encoder_dim, kernel_size=3, padding=1),
                    BaseBlock(encoder_dim),  # TODO: one or two base blocks?
                    BaseBlock(encoder_dim),
                    AttentionBlock(encoder_dim),
                )
            )
        self._spectral_encoders = nn.ModuleList(spectral_encoders)

        decoders = []
        for i, decoder_dim in enumerate(decoder_dims):
            input_dim = encoder_dims[-1] if i == 0 else decoder_dims[i - 1] + encoder_dims[-i - 1]
            decoders.append(
                nn.Sequential(
                    nn.Conv2d(input_dim, decoder_dim, kernel_size=3, padding=1),
                    BaseBlock(decoder_dim),
                    BaseBlock(decoder_dim),
                    AttentionBlock(decoder_dim),
                )
            )
        self._decoders = nn.ModuleList(decoders)

        self._inverse_bock = InverseBlock(encoder_dims[0] + decoder_dims[-1], output_channels)
        self._attention_tail = AttentionTail(output_channels)

        if illuminant_head:
            self.illuminant_head = IlluminantHead(in_ch=encoder_dims[-1], hidden=illuminant_hidden)

        if finetune:
            for name, param in self.named_parameters():
                if "_spectral_encoders" not in name:
                    param.requires_grad = False


    def forward(self, rgb: torch.Tensor, spec: torch.Tensor):
        if len(rgb.shape) != 4:
            raise ValueError(f"Expected [B, C, H, W] input, got {rgb.shape}.")
        if len(spec.shape) != 4:
            raise ValueError(f"Expected [B, C, H, W] input, got {spec.shape}.")
        global_residual = rgb

        rgb_encoder_outputs = []
        x = rgb
        for i, encoder in enumerate(self._rgb_encoders):
            x = encoder(x)
            if i != len(self._rgb_encoders) - 1:
                rgb_encoder_outputs.append(x)
                x = F.max_pool2d(x, kernel_size=2)

        spectral_encoder_outputs = []
        y = spec
        for i, encoder in enumerate(self._spectral_encoders):
            y = encoder(y)
            if i != len(self._spectral_encoders) - 1:
                spectral_encoder_outputs.append(y)

        x = x + F.interpolate(y, size=x.shape[2:], mode="bilinear") if y.shape[2:] != x.shape[2:] else x + y

        if hasattr(self, 'illuminant_head'):
            illuminant = self.illuminant_head(x)

        for i, decoder in enumerate(self._decoders):
            x = decoder(x)
            x = F.interpolate(x, scale_factor=2, mode="bilinear")

            rgb_ft = rgb_encoder_outputs.pop()
            spectral_ft = spectral_encoder_outputs.pop()
            ft = rgb_ft + F.interpolate(spectral_ft, size=rgb_ft.shape[2:], mode="bilinear") if spectral_ft.shape[2:] != rgb_ft.shape[2:] else rgb_ft + spectral_ft
            x = torch.cat([x, ft], dim=1)

        x = self._inverse_bock(x)
        x = self._attention_tail(x)

        if hasattr(self, 'illuminant_head'):
            return x + global_residual, illuminant

        return x + global_residual