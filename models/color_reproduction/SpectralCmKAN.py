import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable
from typing_extensions import Self
from einops import rearrange

import ipdb

def compute_bspline(x: torch.Tensor, grid: torch.Tensor, k: int):
    """
    For a given grid with G_1 intervals and spline order k, we *recursively* compute
    and evaluate each B_n(x_{ij}). x is a (batch_size, in_dim) and grid is a
    (out_dim, in_dim, # grid points + 2k + 1)

    Returns a (batch_size, out_dim, in_dim, grid_size + k) intermediate tensor to 
    compute sum_i {c_i B_i(x)} with.

    """
    
    grid = grid[None, :, :, :].to(x.device)
    x = x[:, None, :, None].to(x.device)
    
    # Base case: B_{i,0}(x) = 1 if (grid_i <= x <= grid_{i+k}) 0 otherwise
    bases = (x >= grid[:, :, :, :-1]) * (x < grid[:, :, :, 1:])

    # Recurse over spline order j, vectorize over basis function i
    for j in range (1, k + 1):
        n = grid.size(-1) - (j + 1)
        b1 = ((x[:, :, :, :] - grid[:, :, :, :n]) / (grid[:, :, :, j:-1] - grid[:, :, :, :n])) * bases[:, :, :, :-1]
        b2 = ((grid[:, :, :, j+1:] - x[:, :, :, :])  / (grid[:, :, :, j+1:] - grid[:, :, :, 1:n+1])) * bases[:, :, :, 1:]
        bases = b1 + b2

    return bases

def generate_control_points(low_bound: float, up_bound: float, in_dim: int,
                            out_dim: int, spline_order: int, grid_size: int):
    """
    Generate a vector of {grid_size} equally spaced points in the interval [low_bound, up_bound] and broadcast (out_dim, in_dim) copies.
    To account for B-splines of order k, using the same spacing, generate an additional
    k points on each side of the interval. See 2.4 in original paper for details.
    """

    # vector of size [grid_size + 2 * spline_order + 1]
    spacing = (up_bound - low_bound) / grid_size
    grid = torch.arange(-spline_order, grid_size + spline_order + 1)
    grid = grid * spacing + low_bound

    # [out_dim, in_dim, G + 2k + 1]
    grid = grid[None, None, ...].expand(out_dim, in_dim, -1).contiguous()
    return grid

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class DWTForward(nn.Module):
    def __init__(self):
        super(DWTForward, self).__init__()
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                            hl[None,::-1,::-1], hh[None,::-1,::-1]],
                            axis=0)
        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad=False)
    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight,] * C, dim=0)
        y = F.conv2d(x, filters, groups=C, stride=2)
        return y

class SpectralEncoder2D(nn.Module):
    def __init__(self, in_dim=15, base_dim=16):
        super().__init__()
        # 64x64 -> 32x32
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim, base_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.down = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, 3, 2, 1),
            nn.ReLU(inplace=True)
        )
        # optional global descriptor
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_dim * 2, base_dim)
    def forward(self, x):
        f1 = self.conv1(x)            # (B, base_dim, 64, 64)
        f2 = self.down(f1)            # (B, 2*base_dim, 32, 32)
        g  = self.pool(f2).flatten(1) # (B, 2*base_dim)
        g  = self.fc(g)               # (B, base_dim)
        return f1, f2, g

class IlluminationEstimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=4, n_fea_out=3):
        super(IlluminationEstimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w
        
        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w
        
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)
        return illu_fea, illu_map
    
class LayerNorm(nn.Module):

    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature_a = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature_v = nn.Parameter(torch.ones(num_heads, 1, 1))

        # q: what we want to attend to (spatial information)
        self.q_proj = nn.Conv2d(
            dim, dim, kernel_size=3,
            padding=1, stride=2, padding_mode='reflect', 
            groups=dim, bias=bias
        )
        # k: what we use to calculate attention (channel information)
        self.k_proj = nn.Conv2d(
            dim, dim, kernel_size=3,
            padding=1, stride=2, padding_mode='reflect', 
            bias=bias
        )
        # v: what we use to calculate the output (channel information)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # a: anchor information (reduced spatial information and channel information)
        self.a_proj = nn.Sequential(
            nn.Conv2d(
                dim, dim, kernel_size=3,
                padding=1, stride=2, padding_mode='reflect', 
                groups=dim, bias=bias
            ),
            nn.Conv2d(dim, dim//2, kernel_size=1)
        )
        # output projection
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, illu_feat):
        b, c, h, w = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x) * illu_feat
        a = self.a_proj(x)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        a = rearrange(a, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        a = torch.nn.functional.normalize(a, dim=-1)

        # Q - C×(H/s×W/s), K - C×(H/s×W/s), V - C×(H×W), A - C/r×(H/s×W/s) 

        # transposed self-attention with attention map of shape (C×C)
        attn_a = (q @ a.transpose(-2, -1)) * self.temperature_a
        attn_a = attn_a.softmax(dim=-1)

        attn_k = (a @ k.transpose(-2, -1)) * self.temperature_v
        attn_k = attn_k.softmax(dim=-1)
        
        out_v = (attn_k @ v)

        out = (attn_a @ out_v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FFN(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden_features,hidden_features, kernel_size=3,stride=1,padding=1,dilation=1,groups=hidden_features)
        self.pointwise2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act_layer = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x

class TransformerBlock(nn.Module):
    """
    from restormer
    input size: (B,C,H,W)
    output size: (B,C,H,W)
    H, W could be different
    """

    def __init__(self, in_channel, mid_channel, out_channel, num_heads, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(in_channel)
        self.attn = Attention(in_channel, num_heads, bias)
        self.norm2 = LayerNorm(in_channel)
        self.ffn = FFN(in_channel, mid_channel, out_channel)

    def forward(self, x, illu_feat):
        x = x + self.attn(self.norm1(x), illu_feat)
        x = x + self.ffn(self.norm2(x))

        return x

class Encoder2D(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(self, in_dim, out_dim, kernel_size):
        super(Encoder2D, self).__init__()
        self.estimator = IlluminationEstimator(12, in_dim+1, in_dim)
        
        self.down1 = DWTForward() # 12 h/2
        self.trans1 = TransformerBlock(
            12, 12, 12, 3, True
        )
        self.illu_down1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(12, 12, 1),
        )

        self.down2 = DWTForward() # 48 h/4
        self.trans2 = TransformerBlock(
            48, 48, 48, 3, True
        )
        self.illu_down2 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(12, 48, 1),
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear')
        
        self.conv_out = nn.Sequential(
            LayerNorm(3+12+48),
            FFN(3+12+48, out_dim)
        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        illu_fea, illu_map = self.estimator(x)
        x = x * illu_map + x # 3 h w

        x1 = self.down1(x)
        illu_fea = self.illu_down1(illu_fea)
        x1 = self.trans1(x1, illu_fea) # 12 h/2 w/2

        x2 = self.down2(x1)
        illu_fea = self.illu_down2(illu_fea)
        x2 = self.trans2(x2, illu_fea) # 48 h/4 w/4

        x1 = self.up1(x1)
        x2 = self.up2(x2)
        x = torch.cat([x, x1, x2], dim=1)
        x = self.conv_out(x)
        return x

class GeneratorLayer(torch.nn.Module):
    """
    sepconv replace conv_out to reduce GFLOPS
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        MID_CHANNELS = 21 * in_channels
        self.encoder = Encoder2D(in_channels, MID_CHANNELS, 3)

        self.norm1 = LayerNorm(MID_CHANNELS)

        N = 128
        self.basis = nn.Parameter(torch.rand(1, MID_CHANNELS, N))

        self.q = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.k = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.v = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)

        self.norm2 = LayerNorm(MID_CHANNELS)

        self.conv_reproj = FFN(in_features=MID_CHANNELS, out_features=out_channels)

    def forward(self, x:torch.Tensor):

        B, C, H, W = x.shape

        # forward projection
        x = self.encoder(x)

        x = self.norm1(x)

        # basis coeff
        q = self.q(x)
        k = self.k(self.basis)
        v = self.v(self.basis)

        q = rearrange(q, 'b c h w -> b c (h w)')

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        a = (q.transpose(-2, -1) @ k).transpose(-2, -1)
        a = F.relu(a)

        y = v @ a

        y = rearrange(y, 'b c (h w) -> b c h w', h=H, w=W)

        # back projection
        x = self.norm2(y)
        x = self.conv_reproj(x)

        return x

class LightEncoder2D(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(self, in_dim, out_dim, kernel_size):
        super(LightEncoder2D, self).__init__()
        self.estimator = IlluminationEstimator(12, in_dim+1, in_dim)
        
        self.down1 = DWTForward() # 12 h/2
        self.trans1 = TransformerBlock(
            12, 12, 12, 3, True
        )
        self.illu_down1 = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(12, 12, 1),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear')
        
        self.conv_out = nn.Sequential(
            LayerNorm(3+12),
            FFN(3+12, out_dim)
        )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        illu_fea, illu_map = self.estimator(x)
        x = x * illu_map + x # 3 h w

        x1 = self.down1(x)
        illu_fea = self.illu_down1(illu_fea)
        x1 = self.trans1(x1, illu_fea) # 12 h/2 w/2

        x1 = self.up1(x1)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_out(x)
        return x

class SpectralLightEncoder2D(LightEncoder2D):
    def __init__(self, in_dim_rgb=3, in_dim_spec=15, out_dim=15):
        super().__init__(in_dim_rgb, out_dim, 3)
        self.spec_encoder = SpectralEncoder2D(in_dim=in_dim_spec, base_dim=16)
        self.spec_proj1 = nn.Conv2d(16, 12, 1)
        self.spec_proj2 = nn.Conv2d(32, 12, 1)
        self.spec_gate = nn.Linear(16, 12)  # for global modulation

    def forward(self, rgb, spectral):
        spec64, spec32, gspec = self.spec_encoder(spectral)
        illu_fea, illu_map = self.estimator(rgb)

        # Inject global spectral context into illumination feature
        gmod = self.spec_gate(gspec).unsqueeze(-1).unsqueeze(-1)
        illu_fea = illu_fea + gmod

        x = rgb * illu_map + rgb
        x1 = self.down1(x)
        illu_fea = self.illu_down1(illu_fea)

        # fuse 64x64 spectral with 256x256 RGB-downsampled (upsample spectral)
        spec_fused = F.interpolate(self.spec_proj1(spec64), size=x1.shape[-2:], mode='bilinear')
        x1 = self.trans1(x1 + spec_fused, illu_fea)

        x1 = self.up1(x1)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_out(x)
        return x


class LightGeneratorLayer(torch.nn.Module):
    """
    sepconv replace conv_out to reduce GFLOPS
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        MID_CHANNELS = 3+12
        self.encoder = LightEncoder2D(in_channels, MID_CHANNELS, 3)

        self.norm1 = LayerNorm(MID_CHANNELS)

        N = 30
        self.basis = nn.Parameter(torch.rand(1, MID_CHANNELS, N))

        self.q = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.k = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)
        self.v = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, kernel_size=1)

        self.norm2 = LayerNorm(MID_CHANNELS)

        self.conv_reproj = FFN(in_features=MID_CHANNELS, out_features=out_channels)

    def forward(self, x:torch.Tensor):

        B, C, H, W = x.shape

        # forward projection
        x = self.encoder(x)

        x = self.norm1(x)

        # basis coeff
        q = self.q(x)
        k = self.k(self.basis)
        v = self.v(self.basis)

        q = rearrange(q, 'b c h w -> b c (h w)')

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        a = (q.transpose(-2, -1) @ k).transpose(-2, -1)
        a = F.relu(a)

        y = v @ a

        y = rearrange(y, 'b c (h w) -> b c h w', h=H, w=W)

        # back projection
        x = self.norm2(y)
        x = self.conv_reproj(x)

        return x

class SpectralLightGeneratorLayer(LightGeneratorLayer):
    def __init__(self, in_channels_rgb=3, in_channels_spec=15, out_channels=90):
        super(LightGeneratorLayer, self).__init__()
        MID_CHANNELS = 3 + 12
        self.encoder = SpectralLightEncoder2D(in_dim_rgb=in_channels_rgb, in_dim_spec=in_channels_spec, out_dim=MID_CHANNELS)
        self.norm1 = LayerNorm(MID_CHANNELS)
        N = 30
        self.basis = nn.Parameter(torch.rand(1, MID_CHANNELS, N))
        self.q = nn.Conv2d(MID_CHANNELS, MID_CHANNELS, 1)
        self.k = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, 1)
        self.v = nn.Conv1d(MID_CHANNELS, MID_CHANNELS, 1)
        self.norm2 = LayerNorm(MID_CHANNELS)
        self.conv_reproj = FFN(MID_CHANNELS, out_features=out_channels)

    def forward(self, rgb, spectral):
        B, C, H, W = rgb.shape
        x = self.encoder(rgb, spectral)
        x = self.norm1(x)
        q = self.q(x)
        k = self.k(self.basis)
        v = self.v(self.basis)
        q = rearrange(q, 'b c h w -> b c (h w)')
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        a = F.relu((q.transpose(-2, -1) @ k).transpose(-2, -1))
        y = v @ a
        y = rearrange(y, 'b c (h w) -> b c h w', h=H, w=W)
        x = self.norm2(y)
        x = self.conv_reproj(x)
        return x

class KANActivation:
    """
    Defines a KAN Activation layer that computes the spline(x) logic
    described in the original paper.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        spline_order: int,
        grid_size: int,
        grid_range: List[float],
    ):
        super(KANActivation, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spline_order = spline_order
        self.grid_size = grid_size
        self.grid_range = grid_range

        self.coef_shape =(out_dim, in_dim, grid_size + spline_order)

        # Generate (out, in) copies of equally spaced control points on [a, b]
        self.grid = generate_control_points(
            grid_range[0],
            grid_range[1],
            in_dim,
            out_dim,
            spline_order,
            grid_size,
        )

        # Define the univariate B-spline function
        self.univarate_fn = compute_bspline

    def __call__(self, x: torch.Tensor, coef) -> torch.Tensor:
        """
        Compute and evaluate the learnable activation functions
        applied to a batch of inputs of size in_dim each.
        """
        grid = self.grid.to(x.device)

        # [bsz x in_dim] to [bsz x out_dim x in_dim x (grid_size + spline_order)]
        bases = self.univarate_fn(x, grid, self.spline_order)

        # [bsz x out_dim x in_dim x (grid_size + spline_order)]
        postacts = bases * coef[None, ...]

        # [bsz x out_dim x in_dim] to [bsz x out_dim]
        spline = torch.sum(postacts, dim=-1)

        return spline

class WeightedResidualLayer:
    """
    Defines the activation function used in the paper,
    phi(x) = w_b SiLU(x) + w_s B_spline(x)
    as a layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        residual_std: float = 0.1,
    ):
        super(WeightedResidualLayer, self).__init__()
        # Residual activation functions
        self.residual_fn = F.silu
        self.univariate_weight_shape = (out_dim, in_dim)
        self.residual_weight_shape = (out_dim, in_dim)

    def __call__(self, x: torch.Tensor, post_acts: torch.Tensor,
                 univariate_weight, residual_weight) -> torch.Tensor:
        """
        Given the input to a KAN layer and the activation (e.g. spline(x)),
        compute a weighted residual.

        x has shape (bsz, in_dim) and act has shape (bsz, out_dim, in_dim)
        """

        # Broadcast the input along out_dim of post_acts
        res = residual_weight * self.residual_fn(x[:, None, :])
        act = univariate_weight * post_acts
        return res + act

class KANLayer:
    "Defines a KAN layer from in_dim variables to out_dim variables."

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grid_size: int,
        spline_order: int,
        residual_std: float = 0.1,
        grid_range: List[float] = [-1, 1],
    ):
        super(KANLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Define univariate function (splines in original KAN)
        self.activation_fn = KANActivation(
            in_dim,
            out_dim,
            spline_order,
            grid_size,
            grid_range,
        )

        # Define the residual connection layer used to compute \phi
        self.residual_layer = WeightedResidualLayer(in_dim, out_dim,
                                                    residual_std)

    def __call__(self, x: torch.Tensor, coef, univariate_weight,
                 residual_weight) -> torch.Tensor:

        spline = self.activation_fn(x, coef)
        phi = self.residual_layer(x, spline, univariate_weight,
                                  residual_weight)

        out = torch.sum(phi, dim=-1)

        return out

class CmKANLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, grid_size, spline_order,
                 residual_std, grid_range):
        super(CmKANLayer, self).__init__()

        self.kan_layer = KANLayer(in_dim=in_channels,
                                  out_dim=out_channels,
                                  grid_size=grid_size,
                                  spline_order=spline_order,
                                  residual_std=residual_std,
                                  grid_range=grid_range)

        # Arbitrary layers configuration fc
        self.kan_params_num = 0
        self.kan_params_indices = [0]

        coef_len = np.prod(self.kan_layer.activation_fn.coef_shape)
        univariate_weight_len = np.prod(
            self.kan_layer.residual_layer.univariate_weight_shape)
        residual_weight_len = np.prod(
            self.kan_layer.residual_layer.residual_weight_shape)
        self.kan_params_indices.extend(
            [coef_len, univariate_weight_len, residual_weight_len])

        self.kan_params_num = np.sum(self.kan_params_indices)
        self.kan_params_indices = np.cumsum(self.kan_params_indices)

        self.generator = GeneratorLayer(in_channels, self.kan_params_num)

    def kan(self, x, w):

        i, j = self.kan_params_indices[0], self.kan_params_indices[1]
        coef = w[:, i:j].view(-1, *self.kan_layer.activation_fn.coef_shape)
        i, j = self.kan_params_indices[1], self.kan_params_indices[2]
        univariate_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.univariate_weight_shape)
        i, j = self.kan_params_indices[2], self.kan_params_indices[3]
        residual_weight = w[:, i:j].view(
            -1, *self.kan_layer.residual_layer.residual_weight_shape)
        x = self.kan_layer(x, coef, univariate_weight, residual_weight)

        return x.squeeze(0)

    def forward(self, x):

        B, C, H, W = x.shape
        
        # kan weights (b, kan_params_num, h, w)
        weights = self.generator(x)
        # kan weights (b, h * w, kan_params_num)
        weights = weights.permute(0, 2, 3, 1)
        weights = weights.reshape(B * H * W, self.kan_params_num)

        x = x.permute(0, 2, 3, 1).reshape(B * H * W, C)

        # img (b * h * w, 3), weights (b * h * w, kan_params_num)
        x = self.kan(x, weights)

        x = x.view(B, H, W, self.kan_layer.out_dim).permute(0, 3, 1, 2)

        return x
    
class LightCmKANLayer(CmKANLayer):
    def __init__(self, in_channels, out_channels, grid_size, spline_order,
                 residual_std, grid_range):
        super(LightCmKANLayer, self).__init__(in_channels, out_channels, grid_size, spline_order,
                 residual_std, grid_range)
        self.generator = LightGeneratorLayer(in_channels, self.kan_params_num)


class SpectralLightCmKANLayer(LightCmKANLayer):
    def __init__(self, in_channels_rgb=3, in_channels_spec=15, out_channels=3, grid_size=5, spline_order=3, residual_std=0.1, grid_range=[0.0,1.0]):
        super(LightCmKANLayer, self).__init__(in_channels_rgb, out_channels, grid_size, spline_order, residual_std, grid_range)
        self.generator = SpectralLightGeneratorLayer(in_channels_rgb, in_channels_spec, self.kan_params_num)

    def forward(self, rgb, spectral):
        B, C, H, W = rgb.shape
        weights = self.generator(rgb, spectral)
        weights = weights.permute(0, 2, 3, 1).reshape(B * H * W, self.kan_params_num)
        x = rgb.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x = self.kan(x, weights)
        x = x.view(B, H, W, self.kan_layer.out_dim).permute(0, 3, 1, 2)
        return x

class CmKAN(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(self, in_dims=[3], out_dims=[3], grid_size=5, spline_order=3, residual_std=0.1, grid_range=[0.0,1.0]):
        super(CmKAN, self).__init__()

        cm_kan_size = [s for s in zip(in_dims, out_dims)]

        self.layers = []
        for in_dim, out_dim in cm_kan_size:
            self.layers.append(
                CmKANLayer(in_channels=in_dim,
                         out_channels=out_dim,
                         grid_size=grid_size,
                         spline_order=spline_order,
                         residual_std=residual_std,
                         grid_range=grid_range))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class LightCmKAN(torch.nn.Module):
    """ Input features BxCxN """

    def __init__(self, in_dims=[3], out_dims=[3], grid_size=5, spline_order=3, residual_std=0.1, grid_range=[0.0,1.0]):
        super(LightCmKAN, self).__init__()

        cm_kan_size = [s for s in zip(in_dims, out_dims)]

        self.layers = []
        for in_dim, out_dim in cm_kan_size:
            self.layers.append(
                LightCmKANLayer(in_channels=in_dim,
                         out_channels=out_dim,
                         grid_size=grid_size,
                         spline_order=spline_order,
                         residual_std=residual_std,
                         grid_range=grid_range))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class SpectralLightCmKAN(nn.Module):
    def __init__(self, in_dims=[3], in_spec=15, out_dims=[3], finetune=False, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            SpectralLightCmKANLayer(in_channels_rgb=in_dim, in_channels_spec=in_spec, out_channels=out_dim, **kwargs)
            for in_dim, out_dim in zip(in_dims, out_dims)
        ])

        # If arg finetune is passed True, freeze all weights that don't belong to SpectralEncoder2D
        if finetune:
            for name, param in self.named_parameters():
                # Only keep parameters in SpectralEncoder2D trainable
                if "spec_encoder" not in name:
                    param.requires_grad = False



    def forward(self, rgb, spectral):
        for layer in self.layers:
            rgb = layer(rgb, spectral)
        return rgb


if __name__ == "__main__":
    # Sanity check

    rgb = torch.randn(2, 3, 512, 512)
    spectral = torch.randn(2, 15, 64, 64)
    model = SpectralLightCmKAN()
    ipdb.set_trace()
    out = model(rgb, spectral)
    print(out.shape)  # (2, 3, 512, 512)