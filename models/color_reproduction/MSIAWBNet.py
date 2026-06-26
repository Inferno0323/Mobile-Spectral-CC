"""
End-to-end AWB network that fuses a high-res rawRGB (3x512x512) and a low-res multispectral image (15x64x64).

Design decisions shown with intermediate tensor shapes and a PyTorch implementation.

High-level structure:
- RGB encoder: downsample 512 -> 256 -> 128 -> 64 (channels 32,64,128,256)
- MSI encoder: keep spatial 64x64 as input spatial size, project to same channel dim as RGB bottleneck (256)
- Cross-attention fusion block at bottleneck (operate on tokens of spatial 64x64 = 4096)
  - Queries from RGB bottleneck, Keys/Values from MSI bottleneck (and optionally reverse attention)
  - MLP and residuals
- Decoder: upsample 64 -> 128 -> 256 -> 512, use skip connections from RGB encoder
- Final conv -> 3 channels (same spatial 512x512)

Shapes (example for batch size B):
- rgb_in: (B, 3, 512, 512)
- msi_in: (B, 15, 64, 64)

RGB encoder outputs (skips):
- e0: (B, 32, 512, 512)
- e1: (B, 64, 256, 256)
- e2: (B,128, 128, 128)
- e3: (B,256,  64,  64)  <- bottleneck from RGB

MSI encoder bottleneck:
- m_b: (B,256,64,64)  <- projected to match e3

Fusion at bottleneck (before decoder):
- flatten spatial: (B, 4096, 256) then transpose for nn.MultiheadAttention which expects (S, N, E): (4096, B, 256)
- cross-attention yields fused tokens -> reshape to (B,256,64,64)

Decoder:
- d3: (B,256,64,64)  (start fused)
- up to (B,128,128,128) + skip e2 -> conv -> (B,128,128,128)
- up to (B,64,256,256) + skip e1 -> conv -> (B,64,256,256)
- up to (B,32,512,512) + skip e0 -> conv -> (B,32,512,512)
- final conv -> (B,3,512,512)

The implementation below is intentionally modular so you can swap blocks (e.g. replace convs by residual blocks or add more attention layers).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Basic building blocks -----------------------------
class ConvBlock(nn.Module):
    """Simple convolution block: Conv -> BN -> GELU (or ReLU) -> Conv -> BN -> GELU"""
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # downsample by 2
        self.pool = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.pool(x)

class UpsampleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # nearest upsample then conv
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

# ----------------------------- Encoders -----------------------------
class RGBEncoder(nn.Module):
    """Encode 3x512x512 -> multi-scale features. Returns skips and final bottleneck."""
    def __init__(self, base_ch=32):
        super().__init__()
        self.enc0 = ConvBlock(3, base_ch)
        self.down1 = Downsample(base_ch, base_ch*2)   # 512->256
        self.enc1 = ConvBlock(base_ch*2, base_ch*2)
        self.down2 = Downsample(base_ch*2, base_ch*4) # 256->128
        self.enc2 = ConvBlock(base_ch*4, base_ch*4)
        self.down3 = Downsample(base_ch*4, base_ch*8) # 128->64
        self.enc3 = ConvBlock(base_ch*8, base_ch*8)

    def forward(self, x):
        e0 = self.enc0(x)        # B,base,512,512
        e1 = self.enc1(self.down1(e0)) # B,base*2,256,256
        e2 = self.enc2(self.down2(e1)) # B,base*4,128,128
        e3 = self.enc3(self.down3(e2)) # B,base*8,64,64
        return [e0, e1, e2, e3]

class MSLEncoder(nn.Module):
    """Encode 15x64x64 -> produce bottleneck with same spatial and channels as RGB bottleneck."""
    def __init__(self, in_ch=15, target_ch=256):
        super().__init__()
        # keep spatial the same (64x64) but increase channels to match target
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, target_ch//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(target_ch//2),
            nn.GELU(),
            nn.Conv2d(target_ch//2, target_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(target_ch),
            nn.GELU(),
        )
    def forward(self, x):
        return self.proj(x)  # B, target_ch, 64,64

# ----------------------------- Cross-attention Fusion -----------------------------
class CrossAttentionFusion(nn.Module):
    """Cross-attention where queries come from rgb tokens, keys/values from msi tokens.
    Both inputs must have same spatial resolution (H=W=64) and same channel dim.
    """
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # use PyTorch MultiheadAttention (expects seq_len, batch, embed_dim)
        self.mha_qk = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)
        # optionally add a reverse cross-attention (msi queries rgb keys) - uncomment to use
        self.mha_kq = None
        # feedforward
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Linear(dim*4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, rgb_feat, msi_feat):
        # both: B, C, H, W
        B, C, H, W = rgb_feat.shape
        S = H*W

        # flatten spatial -> (S, B, C)
        rgb_tokens = rgb_feat.view(B, C, S).permute(2,0,1).contiguous()
        msi_tokens = msi_feat.view(B, C, S).permute(2,0,1).contiguous()

        # Cross attention: Q = rgb, K/V = msi
        rgb_norm = self.norm1(rgb_tokens)
        msi_norm = self.norm1(msi_tokens)
        attn_out, attn_weights = self.mha_qk(rgb_norm, msi_norm, msi_norm, need_weights=False)
        # residual
        rgb_tokens = rgb_tokens + attn_out
        # FFN
        rgb_tokens = rgb_tokens + self.ffn(self.norm2(rgb_tokens))

        # Optionally produce MSI-updated tokens by attending back (not used by default)
        # if self.mha_kq is not None:
        #     msi_out, _ = self.mha_kq(msi_norm, rgb_norm, rgb_norm)
        #     msi_tokens = msi_tokens + msi_out
        #     msi_tokens = msi_tokens + self.ffn(self.norm2(msi_tokens))

        # reshape back to (B, C, H, W)
        fused = rgb_tokens.permute(1,2,0).contiguous().view(B, C, H, W)
        return fused

# ----------------------------- Decoder -----------------------------
class Decoder(nn.Module):
    def __init__(self, base_ch=32):
        super().__init__()
        # d3 input is fused (B, base*8, 64,64)
        self.up2 = UpsampleConv(base_ch*8, base_ch*4)  # -> (B,128,128,128)
        self.dec2 = ConvBlock(base_ch*4 + base_ch*4, base_ch*4) # concat skip e2
        self.up1 = UpsampleConv(base_ch*4, base_ch*2)  # -> (B,64,256,256)
        self.dec1 = ConvBlock(base_ch*2 + base_ch*2, base_ch*2) # concat skip e1
        self.up0 = UpsampleConv(base_ch*2, base_ch)    # -> (B,32,512,512)
        self.dec0 = ConvBlock(base_ch + base_ch, base_ch) # concat skip e0
        self.final_conv = nn.Conv2d(base_ch, 3, kernel_size=1)

    def forward(self, fused, skips):
        # skips: [e0, e1, e2, e3]
        e0, e1, e2, e3 = skips
        x = fused
        x = self.up2(x)            # B,128,128,128
        x = torch.cat([x, e2], dim=1)
        x = self.dec2(x)           # B,128,128,128
        x = self.up1(x)            # B,64,256,256
        x = torch.cat([x, e1], dim=1)
        x = self.dec1(x)           # B,64,256,256
        x = self.up0(x)            # B,32,512,512
        x = torch.cat([x, e0], dim=1)
        x = self.dec0(x)           # B,32,512,512
        out = self.final_conv(x)   # B,3,512,512
        return out

# ---------------------------- Illuminant Estimation Branch ----------------------------
class IlluminantHead(nn.Module):
    def __init__(self, in_ch, hidden=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # global average pooling
        self.fc = nn.Sequential(
            nn.Linear(in_ch, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )
    def forward(self, x):
        x = self.gap(x).squeeze(-1).squeeze(-1)  # B, C
        illum = self.fc(x)  # B, 3
        return illum


# ----------------------------- Full model -----------------------------
class MSIAWBNet(nn.Module):
    def __init__(self, base_ch=32, msi_in_ch=15, num_heads=8, illum_head=False, illum_hidden=128):
        super().__init__()
        self.rgb_enc = RGBEncoder(base_ch=base_ch)
        target_bottleneck_ch = base_ch * 8
        self.msi_enc = MSLEncoder(in_ch=msi_in_ch, target_ch=target_bottleneck_ch)
        self.fusion = CrossAttentionFusion(dim=target_bottleneck_ch, num_heads=num_heads)
        self.decoder = Decoder(base_ch=base_ch)

        if illum_head:
            self.illum_head = IlluminantHead(in_ch=target_bottleneck_ch, hidden=illum_hidden)


    def forward(self, rgb, msi):
        # rgb: B,3,512,512
        # msi: B,15,64,64
        skips = self.rgb_enc(rgb)   # e0,e1,e2,e3
        e3 = skips[-1]              # B,256,64,64 (if base_ch=32)
        m_b = self.msi_enc(msi)     # B,256,64,64
        fused = self.fusion(e3, m_b)
        out = self.decoder(fused, skips)

        if hasattr(self, 'illum_head'):
            illum = self.illum_head(fused)
            return out, illum

        return out
