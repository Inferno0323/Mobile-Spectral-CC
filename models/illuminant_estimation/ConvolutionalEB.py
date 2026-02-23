import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_any_t
from torch.nn.modules import utils
import torchvision.transforms.functional as FF
from torch import Tensor
from PIL import Image
import math
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
import os
from typing import Optional


"""
Parameters for different Edge Based color constancy methods:

WP (white patch)
--njet=0 --mink-norm=-1 --sigma=0

GW (grey world)
--njet=0 --mink-norm=1 --sigma=0

SoG (shades of grey)
--njet=0 --mink-norm=5 --sigma=0

GGW (general grey world)
--njet=0 --mink-norm=5 --sigma=2

GE1 (grey edge 1st order)
--njet=1 --mink-norm=5 --sigma=2

GE2 (grey edge 2nd order)
--njet=2 --mink-norm=5 --sigma=2
"""

class AdaLearnLPP3(nn.Module):
    # AdaptiveLearnableLPPool2d
    def __init__(self, norm_type: float, stride: Optional[_size_any_t] = None, ceil_mode: bool = False, kernel_size: int = 0) -> None:
        super(AdaLearnLPP3, self).__init__()

        self.kernel_size = kernel_size

        self.norm_type = torch.nn.Parameter(torch.tensor(norm_type, dtype=torch.float32))
        self.stride = stride
        self.ceil_mode = ceil_mode

    def forward(self, input: Tensor) -> Tensor:
        if self.kernel_size > 0:
            kernel_size = [self.kernel_size, self.kernel_size]
        else:
            kernel_size = input.shape[2:4]

        kw, kh = utils._pair(kernel_size)


        if self.stride is not None:
            out = F.avg_pool2d(torch.pow(input, self.norm_type), kernel_size, self.stride, 0, self.ceil_mode)
        else:
            out = F.avg_pool2d(torch.pow(input, self.norm_type), kernel_size, padding=0, ceil_mode=self.ceil_mode)

        return torch.pow((torch.sign(out) * F.relu(torch.abs(out))).mul(kw * kh), 1.0 / self.norm_type)


class ConvolutionalEB(nn.Module):
    def __init__(self, njet=0, sigma=None, mink_norm=1.0):
        super(ConvolutionalEB, self).__init__()

        # Traditional parameters from Low-Level Color Constancy
        #  njet: derivative filter order (n)
        #  sigma: gaussian standard deviation (s)
        #  mink_norm: minkoski norm (p)

        W = (np.floor(3*sigma+0.5)*2+1).astype(int).item()
        
        # Initialize convolutional filters
        # Conv2D-1
        if sigma >= 0:
            padding_mode = 'replicate'
        else:
            padding_mode = 'zeros'
        self.conv1 = nn.Conv2d(3, 9, W, stride=1, padding=math.floor(((W-1)/2)+0.5), padding_mode=padding_mode)
        
        # Conv2D-2
        self.conv2 = nn.Conv2d(9, 3, 1, stride=1, padding=0)
        self.conv2.bias.data.fill_(1.)

        
        # .weight shape: # (out_channels, in_channels, kernel_size[0], kernel_size[1])
        # .bias shape: # (out_channels)

        # Midpoint for filters of size W. Floor used only to handle even-sized filters.
        midW = np.floor((W-1)/2.).astype(int)

        if sigma == 0:
            # Grey World, White Patch, Shades of Grey

            self.conv1.weight.data.fill_(0.)
            for ii in range(3):
                self.conv1.weight.data[ii, ii, midW, midW] = 1.
            self.conv1.bias.data.fill_(0.)


            self.conv2.weight.data.fill_(0.)
            for ii in range(3):
                self.conv2.weight.data[ii, ii, 0, 0] = 1.
            self.conv2.bias.data.fill_(0.)

        else:
            half_filter_size = torch.floor(torch.tensor(3*sigma+0.5))
            x = torch.arange(-half_filter_size, half_filter_size+1)
            Gauss = 1/(torch.sqrt(2 * torch.tensor(np.pi)) * sigma)* torch.exp((x**2)/(-2 * sigma * sigma))

            if njet == 0:
                # General Grey World

                # initialize conv1 as 3 independent gaussian filters with specified sigma ---------------
                G0 = Gauss/torch.sum(Gauss)
                G00 = G0[...,None]*G0[None,...] # gd00 = filter2(G', G, 'full'); filter2(G', filter2(G, f_ggw(:,:,ii)));
                # Crop or center G00
                G00 = FF.center_crop(G00, [W,W])
                self.conv1.weight.data.fill_(0.)

                self.conv1.weight.data[0, 0, :, :] = G00 # gDer(input_data(:,:,ii),sigma,0,0);
                self.conv1.weight.data[1, 1, :, :] = G00 # "
                self.conv1.weight.data[2, 2, :, :] = G00 # "
                self.conv1.bias.data.fill_(0.)


                # initialize conv2 as ... ----------------------------------------------------------
                self.conv2.weight.data.fill_(0.)
                self.conv2.weight.data[0, 0, 0, 0] = 1.
                self.conv2.weight.data[1, 1, 0, 0] = 1.
                self.conv2.weight.data[2, 2, 0, 0] = 1.
                self.conv2.bias.data.fill_(0.)

            elif njet == 1:
                # 1st-order Grey Edge

                # initialize conv1 as six independent gaussian-derivative filters with specified sigma ---------------
                G0 = Gauss/torch.sum(Gauss)
                G1 = -(x/sigma**2)*Gauss
                G1 = G1/(torch.sum(torch.sum(x*G1)))
                G10 = G0[...,None]*G1[None,...] # gd10 = filter2(G0', G1, 'full'); # filter2(G0', filter2(G1, f_ge1(:,:,ii)));
                G01 = G1[...,None]*G0[None,...] # gd01 = -filter2(G1', G0, 'full'); # filter2(G1', filter2(G0, f_ge1(:,:,ii)));
                # Crop or center G10 G01
                G10 = FF.center_crop(G10, [W,W])
                G01 = FF.center_crop(G01, [W,W])
                self.conv1.weight.data.fill_(0.)
                self.conv1.weight.data[0, 0, :, :] = G10 # Rx=gDer(R,sigma,1,0);
                self.conv1.weight.data[1, 1, :, :] = G10 # Gx=gDer(G,sigma,1,0);
                self.conv1.weight.data[2, 2, :, :] = G10 # Bx=gDer(B,sigma,1,0);
                self.conv1.weight.data[3, 0, :, :] = G01 # Ry=gDer(R,sigma,0,1);
                self.conv1.weight.data[4, 1, :, :] = G01 # Gy=gDer(G,sigma,0,1);
                self.conv1.weight.data[5, 2, :, :] = G01 # By=gDer(B,sigma,0,1);
                self.conv1.bias.data.fill_(0.)


                # initialize conv2 as ... ----------------------------------------------------------
                self.conv2.weight.data.fill_(0.)
                # Rx.^2+Ry.^2
                self.conv2.weight.data[0, 0, 0, 0] = 1.
                self.conv2.weight.data[0, 3, 0, 0] = 1.
                # Gx.^2+Gy.^2
                self.conv2.weight.data[1, 1, 0, 0] = 1.
                self.conv2.weight.data[1, 4, 0, 0] = 1.
                # Bx.^2+By.^2
                self.conv2.weight.data[2, 2, 0, 0] = 1.
                self.conv2.weight.data[2, 5, 0, 0] = 1.
                self.conv2.bias.data.fill_(0.)

            elif njet == 2:
                # 2nd-order Grey Edge

                # initialize conv1 as nine independent gaussian-derivative filters with specified sigma ---------------
                G0 = Gauss/torch.sum(Gauss)
                G1 = -(x/sigma**2)*Gauss
                G1 = G1/(torch.sum(torch.sum(x*G1)))
                G2 = (x**2/sigma**4-1/sigma**2)*Gauss
                G2 = G2-torch.sum(G2)/len(x)
                G2 = G2/torch.sum(0.5*x*x*G2)
                G11 = G1[...,None]*G1[None,...] # gd11 = -filter2(G1', G1, 'full'); # filter2(G1', filter2(G1, f_ge2(:,:,ii)));
                G02 = G2[...,None]*G0[None,...] # gd02 = filter2(G2', G0, 'full'); # filter2(G2', filter2(G0, f_ge2(:,:,ii)));
                G20 = G0[...,None]*G2[None,...] # gd20 = filter2(G0', G2, 'full'); # filter2(G0', filter2(G2, f_ge2(:,:,ii)));
                # Crop or center G11 G02 G20
                G11 = FF.center_crop(G11, [W,W])
                G02 = FF.center_crop(G02, [W,W])
                G20 = FF.center_crop(G20, [W,W])
                self.conv1.weight.data.fill_(0.)
                self.conv1.weight.data[0, 0, :, :] = G20 # Rxx=gDer(R,sigma,2,0);
                self.conv1.weight.data[1, 1, :, :] = G20 # Gxx=gDer(G,sigma,2,0);
                self.conv1.weight.data[2, 2, :, :] = G20 # Bxx=gDer(B,sigma,2,0);
                self.conv1.weight.data[3, 0, :, :] = G02 # Ryy=gDer(R,sigma,0,2);
                self.conv1.weight.data[4, 1, :, :] = G02 # Gyy=gDer(G,sigma,0,2);
                self.conv1.weight.data[5, 2, :, :] = G02 # Byy=gDer(B,sigma,0,2);
                self.conv1.weight.data[6, 0, :, :] = G11 # Rxy=gDer(R,sigma,1,1);
                self.conv1.weight.data[7, 1, :, :] = G11 # Gxy=gDer(G,sigma,1,1);
                self.conv1.weight.data[8, 2, :, :] = G11 # Bxy=gDer(B,sigma,1,1);
                self.conv1.bias.data.fill_(0.)


                # initialize conv2 as ... ----------------------------------------------------------
                self.conv2.weight.data.fill_(0.)
                # Rxx.^2+4*Rxy.^2+Ryy.^2
                self.conv2.weight.data[0, 0, 0, 0] = 1.
                self.conv2.weight.data[0, 6, 0, 0] = 2.
                self.conv2.weight.data[0, 3, 0, 0] = 1.
                # Gxx.^2+4*Gxy.^2+Gyy.^2
                self.conv2.weight.data[1, 1, 0, 0] = 1.
                self.conv2.weight.data[1, 7, 0, 0] = 2.
                self.conv2.weight.data[1, 4, 0, 0] = 1.
                # Bxx.^2+4*Bxy.^2+Byy.^2
                self.conv2.weight.data[2, 2, 0, 0] = 1.
                self.conv2.weight.data[2, 8, 0, 0] = 2.
                self.conv2.weight.data[2, 5, 0, 0] = 1.
                self.conv2.bias.data.fill_(0.)

            else:
                error('Unsupported njet > 2')

        
        if mink_norm > 0:
            self.poolN = AdaLearnLPP3(norm_type=mink_norm)
        elif mink_norm == -1:
            self.poolN = nn.AdaptiveMaxPool2d(1)
            # TODO: replace with nn.MaxPool2d to handle local max pooling, for spatially varying estimation
        else:
            self.poolN = nn.Identity()

        if njet < 1:
            self.power = 1
        else:
            self.power = 2

    def forward(self, x):

        x = x*255.0

        x = self.conv1(x)
        x = torch.pow(x, self.power)
        x = self.conv2(x)
        x = torch.pow(x, 1/self.power)
        x = self.poolN(x)

        x = F.normalize(x, p=2, dim=1)

        x = x[:,:,0,0]

        return x


