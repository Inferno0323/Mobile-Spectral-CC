import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from glob import glob
from shutil import copyfile
import itertools

EPS = 1e-9
PI = 22.0 / 7.0


def get_hist_boundary():
    """ Returns histogram boundary values.

    Returns:
    bounardy_values: a list of boundary values.
    """

    boundary_values = [-2.85, 2.85]
    assert (boundary_values[0] == -boundary_values[1])
    return boundary_values

def get_uv_coord(hist_size, tensor=True, normalize=False, device='cpu'):
    """ Gets uv-coordinate extra channels to augment each histogram as
    mentioned in the paper.

    Args:
    hist_size: histogram dimension (scalar).
    tensor: boolean flag for input torch tensor; default is true.
    normalize: boolean flag to normalize each coordinate channel; default
        is false.
    device: output tensor allocation ('cuda' or 'cpu'); default is 'cpu'.

    Returns:
    u_coord: extra channel of the u coordinate values; if tensor arg is True,
        the returned tensor will be in (1 x height x width) format; otherwise,
        it will be in (height x width) format.
    v_coord: extra channel of the v coordinate values. The format is the same
        as for u_coord.
    """

    u_coord, v_coord = np.meshgrid(
    np.arange(-(hist_size - 1) / 2, ((hist_size - 1) / 2) + 1),
    np.arange((hist_size - 1) / 2, (-(hist_size - 1) / 2) - 1, -1))
    if normalize:
        u_coord = (u_coord + ((hist_size - 1) / 2)) / (hist_size - 1)
        v_coord = (v_coord + ((hist_size - 1) / 2)) / (hist_size - 1)
    if tensor:
        u_coord = torch.from_numpy(u_coord).to(device=device, dtype=torch.float32)
        u_coord = torch.unsqueeze(u_coord, dim=0)
        u_coord.requires_grad = False
        v_coord = torch.from_numpy(v_coord).to(device=device, dtype=torch.float32)
        v_coord = torch.unsqueeze(v_coord, dim=0)
        v_coord.requires_grad = False
    return u_coord, v_coord


def from_coord_to_uv(hist_size, u, v):
    """ Calculates the corresponding log-chroma values of given (u,v) coordinates.

    Args:
    hist_size: histogram dimension (scalar).
    u, v: input u,v coordinates.

    Returns:
    corresponding log-chroma values.

    """

    coord_range = get_hist_boundary()
    space_range = -coord_range[0] + coord_range[1]
    scale = space_range / hist_size
    U = u * scale
    V = v * scale
    return U, V


def compute_histogram(chroma_input, hist_boundary, nbins, rgb_input=None):
    """ Computes log-chroma histogram of a given log-chroma values.

    Args:
    chroma_input: k x 2 array of log-chroma values; k is the total number of
        pixels and 2 is for the U and V values.
    hist_boundary: histogram boundaries obtained from the 'get_hist_boundary'
        function.
    nbins: number of histogram bins.
    rgb_input: k x 3 array of rgb colors; k is the totanl number of pixels and
        3 is for the rgb vectors. This is an optional argument, if it is
        omitted, the computed histogram will not consider the overall
        brightness value in Eq. 3 in the paper.

    Returns:
    N: nbins x nbins log-chroma histogram.
    """

    eps = np.sum(np.abs(hist_boundary)) / (nbins - 1)
    hist_boundary = np.sort(hist_boundary)
    A_u = np.arange(hist_boundary[0], hist_boundary[1] + eps / 2, eps)
    A_v = np.flip(A_u)
    if rgb_input is None:
        Iy = np.ones(chroma_input.shape[0])
    else:
        Iy = np.sqrt(np.sum(rgb_input ** 2, axis=1))
    # differences in log_U space
    diff_u = np.abs(np.tile(chroma_input[:, 0], (len(A_u), 1)).transpose() -
                    np.tile(A_u, (len(chroma_input[:, 0]), 1)))

    # differences in log_V space
    diff_v = np.abs(np.tile(chroma_input[:, 1], (len(A_v), 1)).transpose() -
                    np.tile(A_v, (len(chroma_input[:, 1]), 1)))

    # counts only U values that is higher than the threshold value
    diff_u[diff_u > eps] = 0
    diff_u[diff_u != 0] = 1

    # counts only V values that is higher than the threshold value
    diff_v[diff_v > eps] = 0
    diff_v[diff_v != 0] = 1

    Iy_diff_v = np.tile(Iy, (len(A_v), 1)) * diff_v.transpose()
    N = np.matmul(Iy_diff_v, diff_u)
    norm_factor = np.sum(N) + EPS
    N = np.sqrt(N / norm_factor)  # normalization
    return N

def get_hist_colors(img, from_rgb):
    """ Gets valid chroma and color values for histogram computation.

    Args:
    img: input image as an ndarray in the format (height x width x channel).
    from_rgb: a function to convert from rgb to chroma.

    Returns:
    valid_chroma: valid chroma values.
    valid_colors: valid rgb color values.
    """

    img_r = np.reshape(img, (-1, 3))
    img_chroma = from_rgb(img_r)
    valid_pixels = np.sum(img_r, axis=1) > EPS  # exclude any zero pixels
    valid_chroma = img_chroma[valid_pixels, :]
    valid_colors = img_r[valid_pixels, :]
    return valid_chroma, valid_colors

def rgb_to_uv(rgb, tensor=False):
    """ Converts RGB to log-chroma space.

    Args:
      rgb: input color(s) in rgb space.
      tensor: boolean flag for input torch tensor; default is false.

    Returns:
      color(s) in chroma log-chroma space.
    """

    if tensor:
        log_rgb = torch.log(rgb + EPS)
        u = log_rgb[:, 1] - log_rgb[:, 0]
        v = log_rgb[:, 1] - log_rgb[:, 2]
        return torch.stack([u, v], dim=-1)
    else:
        log_rgb = np.log(rgb + EPS)
        u = log_rgb[:, 1] - log_rgb[:, 0]
        v = log_rgb[:, 1] - log_rgb[:, 2]
        return np.stack([u, v], axis=-1)


def uv_to_rgb(uv, tensor=False):
    """ Converts log-chroma space to RGB.

    Args:
        uv: input color(s) in chroma log-chroma space.
        tensor: boolean flag for input torch tensor; default is false.

    Returns:
        color(s) in rgb space.
    """

    if tensor:
        rb = torch.exp(-uv)
        rgb = torch.stack([rb[:, 0], torch.ones(
        rb.shape[0], dtype=uv.dtype, device=uv.device), rb[:, 1]],
                        dim=-1)
        rgb = rgb / torch.unsqueeze(vect_norm(rgb, tensor), dim=-1)
        return rgb
    else:
        rb = np.exp(-uv)
        rgb = np.stack([rb[:, 0], np.ones(rb.shape[0]), rb[:, 1]], axis=-1)
        return rgb / np.transpose(np.tile(vect_norm(rgb), (3, 1)))


def vect_norm(vect, tensor=False, axis=1):
    """ Computes vector norm.

    Args:
        vect: input vector(s) (float).
        tensor: boolean flag for input torch tensor; default is false.
        axis: sum axis; default is 1.

    Returns:
        vector norm.
    """

    if tensor:
        return torch.sqrt(torch.sum(vect ** 2, dim=axis))
    else:
        return np.sqrt(np.sum(vect ** 2, axis=axis))






# ----------------------------
#  HISTOGRAM BUILDER (BATCH)
# ----------------------------
class UVHistogramBatch:
    def __init__(self, nbins=256, device='cpu'):
        self.nbins = nbins
        self.device = device
        self.boundary = get_hist_boundary()
        self.boundary = np.sort(self.boundary)
        self.eps_bin = np.sum(np.abs(self.boundary)) / (nbins - 1)
        # precompute bin edges
        self.u_vals = np.arange(self.boundary[0], self.boundary[1]+self.eps_bin/2, self.eps_bin)
        self.v_vals = np.flip(self.u_vals).copy()
        self.u_vals_t = torch.from_numpy(self.u_vals).float().to(device)
        self.v_vals_t = torch.from_numpy(self.v_vals).float().to(device)

    def image_to_hist(self, chroma, rgb=None):
        """
        chroma: (B,K,2) UV values per pixel
        rgb: (B,K,3) RGB per pixel for brightness weighting (optional)
        Returns N: (B,nbins,nbins)
        """
        B,K,_ = chroma.shape
        nbins = self.nbins
        if rgb is None:
            Iy = torch.ones((B,K), device=chroma.device)
        else:
            Iy = torch.sqrt((rgb**2).sum(dim=2))  # luminance weight

        # compute differences for u and v
        # We’ll do a fast version using broadcasting:
        u_grid = self.u_vals_t[None,None,:]  # 1x1xnbins
        v_grid = self.v_vals_t[None,None,:]
        u_val = chroma[:,:,0][:,:,None]  # BxKx1
        v_val = chroma[:,:,1][:,:,None]

        diff_u = (u_val - u_grid).abs()
        diff_v = (v_val - v_grid).abs()

        mask_u = (diff_u <= self.eps_bin).float()
        mask_v = (diff_v <= self.eps_bin).float()

        Iy_exp_v = Iy[:,:,None] * mask_v  # BxKxnbins
        # matrix multiply Iy_exp_v.T x mask_u to accumulate counts
        # We can do for each batch in a loop (nbins=256 so okay):
        N_out = []
        for b in range(B):
            # (K,nbins) x (K,nbins)
            mat = torch.matmul(Iy_exp_v[b].T, mask_u[b])
            N_out.append(mat)
        N = torch.stack(N_out, dim=0)  # Bxnbinsxnbins
        norm_factor = N.sum(dim=(1,2), keepdim=True) + EPS
        N = torch.sqrt(N / norm_factor)
        return N

# ----------------------------
#  PYRAMID FILTER MODULE
# ----------------------------
class PyramidFilter(nn.Module):
    def __init__(self, nbins, levels=7, small_k=5, lambda_reg=1e-3):
        super().__init__()
        self.nbins = nbins
        self.levels = levels
        self.small_k = small_k
        self.filters = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, small_k, small_k) * 0.01) for _ in range(levels)
        ])
        self.lambda_reg = lambda_reg

    def forward(self, hist):  # hist: (B,1,H,W)
        x = hist
        B = x.shape[0]
        pyramid = [x]
        for l in range(1, self.levels):
            kernel = torch.tensor([[1.,2.,1.],
                                   [2.,4.,2.],
                                   [1.,2.,1.]], device=x.device)
            kernel = (kernel / kernel.sum()).view(1,1,3,3)
            C = x.shape[1]
            krep = kernel.repeat(C,1,1,1)
            blurred = F.conv2d(pyramid[-1], krep, padding=1, groups=C)
            down = F.avg_pool2d(blurred, kernel_size=2, stride=2)
            pyramid.append(down)

        responses = []
        for l, lvl in enumerate(pyramid):
            filt = self.filters[l]
            pad = self.small_k // 2
            resp = F.conv2d(lvl, filt, padding=pad)
            if l > 0:
                resp = F.interpolate(resp, size=(self.nbins,self.nbins), mode='bilinear', align_corners=False)
            responses.append(resp)
        out = sum(responses)  # (B,1,H,W)
        return out

    def l2_regularization(self):
        s = 0.0
        for p in self.filters:
            s = s + torch.sum(p*p)
        return self.lambda_reg * s

class CCC(nn.Module):
    def __init__(self, input_size=256, device='cpu'):
        super().__init__()
        self.input_size = input_size
        self.device = device
        u_coord, v_coord = get_uv_coord(self.input_size,
                                        tensor=True,
                                        device=self.device)
        self.register_buffer("u_coord", u_coord)
        self.register_buffer("v_coord", v_coord)
        # learnable CCC parameters:
        self.F = nn.Parameter(torch.randn(2, input_size, input_size) * 0.01)
        self.B = nn.Parameter(torch.zeros(input_size, input_size))
        # optional gain
        self.G = None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, N):
        """
        N: input histogram(s) in (batch x 2 x H x W)
           (first two channels = main/edge histograms)
        Returns: rgb (batch x 3), P (batch x H x W)
        """
        # Fourier conv like in C5
        N_fft = torch.rfft(N[:, :2, :, :], 2, onesided=False)
        F_fft = torch.rfft(self.F, 2, onesided=False)
        N_after_conv = torch.irfft(
            ops.complex_multiplication(N_fft, F_fft), 2, onesided=False)

        N_after_conv = torch.sum(N_after_conv, dim=1)
        N_after_bias = N_after_conv + self.B
        N_after_bias = torch.clamp(N_after_bias, -100, 100)

        P = self.softmax(torch.reshape(N_after_bias,
                                       (N_after_bias.shape[0], -1)))
        P = torch.reshape(P, N_after_bias.shape)

        u = torch.sum(P * self.u_coord, dim=[-1, -2])
        v = torch.sum(P * self.v_coord, dim=[-1, -2])
        u, v = ops.from_coord_to_uv(self.input_size, u, v)
        rgb = ops.uv_to_rgb(torch.stack([u, v], dim=1), tensor=True)
        return rgb, P
