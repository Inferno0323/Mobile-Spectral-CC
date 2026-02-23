import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from torch.nn.common_types import _size_any_t
from torch.nn.modules import utils

import ipdb

class AdaLearnLPP3(nn.Module):
    # AdaptiveLearnableLPPool2d
    def __init__(self, norm_type: float, pow_protection: str, stride: Optional[_size_any_t] = None, ceil_mode: bool = False, kernel_size: int = 0) -> None:
        super(AdaLearnLPP3, self).__init__()

        self.kernel_size = kernel_size

        self.norm_type = torch.nn.Parameter(torch.tensor(norm_type))
        self.stride = stride
        self.ceil_mode = ceil_mode

        if pow_protection == 'abs':
            self.pow_protection = torch.abs
        elif pow_protection == 'relu':
            self.pow_protection = F.relu
        elif pow_protection == 'posrelu':
            self.pow_protection = PosReLU()
        elif pow_protection == 'onerelu':
            self.pow_protection = PosReLU(offset=1.0)
        elif pow_protection == 'oneabs':
            self.pow_protection = lambda a : torch.abs(a-1)+1
        else:
            self.pow_protection = lambda a : a


    def forward(self, input: Tensor) -> Tensor:
        if self.kernel_size > 0:
            kernel_size = [self.kernel_size, self.kernel_size]
        else:
            kernel_size = input.shape[2:4]

        kw, kh = utils._pair(kernel_size)

        exp = self.pow_protection(self.norm_type)

        if self.stride is not None:
            out = F.avg_pool2d(torch.pow(input, exp), kernel_size, self.stride, 0, self.ceil_mode)
        else:
            out = F.avg_pool2d(torch.pow(input, exp), kernel_size, padding=0, ceil_mode=self.ceil_mode)

        return torch.pow((torch.sign(out) * F.relu(torch.abs(out))).mul(kw * kh), 1.0 / exp)

# Positive ReLU
class PosReLU(nn.Module):
    def __init__(self, offset: float = 0.0000001) -> None:
        super(PosReLU, self).__init__()
        self.offset = offset

    def forward(self, input: Tensor) -> Tensor:
        out = F.relu(input-self.offset)+self.offset
        return out

class ConvMean(nn.Module):
    
    def __init__(self, inp_size=3):
        super(ConvMean, self).__init__()
        
        self.conv1 = nn.Conv2d(inp_size, 7, 3, stride=1, padding=1, padding_mode='replicate')
        # Conv2D (intermediate)
        self.convI = nn.Conv2d(7, 14, 3, stride=1, dilation=1, padding=1)
        
        # Conv2D-2
        self.conv2 = nn.Conv2d(14, 3, 1, stride=1, padding=0)
        self.conv2.bias.data.fill_(1.)

        self.pool1 = nn.MaxPool2d(2, stride=1, padding=1)
        self.poolN = AdaLearnLPP3(norm_type=1.000001, pow_protection="oneabs")
        self.power = 1
        
        self.nl1 = nn.PReLU(init=1.0)
        self.nlI = nn.PReLU(init=1.0)
        self.nl2 = PosReLU(offset=0.00000000000000000001)
        self.mink_protection = torch.abs
        
    def forward(self, x):
        x = x*255.0

        x = self.conv1(x)
        x = self.nl1(x)
        x = self.pool1(x)
        x = torch.pow(x, self.power)

        x = self.convI(x)
        x = self.nlI(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.nl2(x)
        x = self.pool1(x)
        x = torch.pow(x, 1/self.power)

        x = self.mink_protection(x)
        x = self.poolN(x)

        x = F.normalize(x, p=2, dim=1)

        x = x[:,:,0,0]
        
        return x

