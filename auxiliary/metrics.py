import torch
from torch import nn
import torchmetrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
from .color_utils import Illuminant, CMF
from skimage import color
import math


# Losses

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, y_hat, y):
        return self.l1(y_hat, y)

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_hat, y):
        return self.mse(y_hat, y)

class AngularErrorLoss(nn.Module):
    def __init__(self, safe_v = 0.999999):
        super(AngularErrorLoss, self).__init__()
        self.safe_v = safe_v

    def forward(self, y_hat, y):
        if len(y_hat.shape) == 4 and len(y.shape) == 2:
            b, c, h, w = y_hat.shape
            y = y[:,:, None, None].expand_as(y_hat)

        y_hat = y_hat / (torch.norm(y_hat, dim=1, keepdim=True) + 1e-8)
        y = y / (torch.norm(y, dim=1, keepdim=True) + 1e-8)

        cos = torch.sum(y_hat * y, dim=1).clamp(-self.safe_v, self.safe_v)
        angle = torch.acos(cos)

        return torch.mean(angle)

class ChromaticLoss(nn.Module):
    """ 
    Pixelwise angular error between two images
    """
    def __init__(self):
        super(ChromaticLoss, self).__init__()
        self.name = "ChromaticLoss"
        self.safe_v = 0.999999

    def forward(self,  y_hat, y):
        y_hat = y_hat / (torch.norm(y_hat, dim=1, keepdim=True) + 1e-8)
        y = y / (torch.norm(y, dim=1, keepdim=True) + 1e-8)

        cos = torch.sum(y_hat * y, dim=1).clamp(-self.safe_v, self.safe_v)
        
        return (torch.ones_like(cos)-cos).mean()

class SafePowClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, eps=1e-6, max_grad=10.0):
        ctx.save_for_backward(x)
        ctx.p = p
        ctx.eps = eps
        ctx.max_grad = max_grad
        return torch.sign(x) * torch.pow(torch.abs(x) + eps, p)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output * ctx.p * torch.pow(torch.abs(x) + ctx.eps, ctx.p - 1)
        grad_input = torch.clamp(grad_input, -ctx.max_grad, ctx.max_grad)
        return grad_input, None, None, None

# class deltaE76Loss(nn.Module):

#     def __init__(self, input_type="xyz"):
#         super().__init__()
#         self.name = "deltaE76Loss"

#     def safe_pow(self, x, p, eps=1e-6):
#         """Safe power function for gradients"""
#         if p % 1 != 0:  # fractional exponent
#             return torch.sign(x) * torch.pow(torch.abs(x) + eps, p)
#         else:
#             return x ** p    
    
#     def safe_div(self, a, b, eps=1e-6):
#         """Safe division function for gradients"""
#         return a / (b+eps) if b == 0 else a / b

#     # def smooth_cbrt(self,x, eps=1e-6):
#     #     return torch.sign(x) * ((x**2 + eps)**(1/6))

#     def xyz2lab(self, xyz, ill_xyz=torch.tensor([0.95047, 1., 1.08883])):
        
#         x, y, z = xyz[:,0,:,:], xyz[:,1,:,:], xyz[:,2,:,:]
#         X_n, Y_n, Z_n = ill_xyz

#         delta = (6/29)**3


#         def f(t):
#             """Safe f(t) function"""
#             safe_t = torch.clamp(t, min=delta)
#             return torch.where(
#                 t > delta,
#                 # SafePowClip.apply(t, 1/3), # redundant safe pow, but better safe than sorry
#                 self.safe_pow(safe_t, 1/3, eps=1e-4),
#                 t * (29/6)**2 / 3 + (4/29)
#             )
            
#             # linear_mask = (t <= delta).type(torch.FloatTensor).to(t.device)
#             # exponential_mask = (t > delta).type(torch.FloatTensor).to(t.device)
#             # return (self.safe_pow(t, 1/3)) * exponential_mask + (t * (29/6)**2 / 3 + (4/29)) * linear_mask
         
#             # mask = (t > delta).float()
#             # return mask * torch.pow(safe_t, 1/3) + (1 - mask) * (t * (29/6)**2 / 3 + (4/29))


#         # fx, fy, fz = f(x/X_n), f(y/Y_n), f(z/Z_n)
#         fx, fy, fz = f(self.safe_div(x, X_n)), f(self.safe_div(y, Y_n)), f(self.safe_div(z, Z_n))
#         # fx, fy, fz = self.safe_div(x, X_n), self.safe_div(y, Y_n), self.safe_div(z, Z_n)

# 	# convert to lab


#         L = (116 * fy - 16) / 100
#         a = (500 * (fx - fy) + 128) / 255
#         b = (200 * (fy - fz) + 128) / 255

#         Lab = torch.stack([L, a, b], dim=1)
#         return Lab


#     def forward(self, img1: torch.Tensor, img2: torch.Tensor):
#         """
#         img1, img2: torch.Tensor of shape BCHW
#         Returns: mean ΔE76 over all pixels and batch
#         """
#         b, c, h, w = img1.shape
#         # def check_nan_grad(grad):
#         #     # Find indices where the gradient is NaN
#         #     nan_mask = torch.isnan(grad)
#         #     if nan_mask.any():
#         #         print("NaN detected in gradient!")
#         #         print("Indices with NaN:", nan_mask.nonzero())
#         #         print("Corresponding forward values:", img1[nan_mask])
#         #     return grad  # must return grad

#         # img1.register_hook(check_nan_grad)
#         img1_lab = self.xyz2lab(img1)
#         img2_lab = self.xyz2lab(img2)
#         # Flatten pixels
#         arr1 = img1_lab.permute(0, 2, 3, 1).reshape(b, -1, 3)
#         arr2 = img2_lab.permute(0, 2, 3, 1).reshape(b, -1, 3)

#         Lstd, astd, bstd = arr1[...,0], arr1[...,1], arr1[...,2]
#         Lsample, asample, bsample = arr2[...,0], arr2[...,1], arr2[...,2]

#         deltaE = torch.sqrt(self.safe_pow(Lstd - Lsample, 2) + self.safe_pow(astd - asample, 2) + self.safe_pow(bstd - bsample, 2))

#         return deltaE.mean()

class deltaE76Loss(nn.Module):

    def __init__(self, input_type="xyz"):
        super().__init__()
        self.name = "deltaE76Loss"
    
    def xyz2lab(self, xyz, ill_xyz=torch.tensor([0.95047, 1., 1.08883])):
        x, y, z = xyz[:,0,:,:], xyz[:,1,:,:], xyz[:,2,:,:]
        X_n, Y_n, Z_n = ill_xyz

        e = 216/24389
        k = 24389/27

        def cbrt(x):
            """Safe cube root function"""
            return torch.sign(x) * torch.abs(x).pow(1/3)

        def f(t):
            """Safe f(t) function with clamping"""            
            return torch.where(
                t > e,
                cbrt(t), 
                (t * k + 16)/116
            )

        fx, fy, fz = f(x/X_n), f(y/Y_n), f(z/Z_n)

        L = (116 * fy - 16) / 100
        a = (500 * (fx - fy) + 128) / 255
        b = (200 * (fy - fz) + 128) / 255

        Lab = torch.stack([L, a, b], dim=1)
        return Lab


    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        img1, img2: torch.Tensor of shape BCHW
        Returns: mean ΔE76 over all pixels and batch
        """
        b, c, h, w = img1.shape
        img1_lab = self.xyz2lab(img1.clamp_min(1e-6))
        img2_lab = self.xyz2lab(img2)
        # Flatten pixels
        arr1 = img1_lab.permute(0, 2, 3, 1).reshape(b, -1, 3)
        arr2 = img2_lab.permute(0, 2, 3, 1).reshape(b, -1, 3)

        Lstd, astd, bstd = arr1[...,0], arr1[...,1], arr1[...,2]
        Lsample, asample, bsample = arr2[...,0], arr2[...,1], arr2[...,2]

        deltaE = torch.sqrt((Lstd - Lsample)**2 + (astd - asample)**2 + (bstd - bsample)**2 + 1e-8)

        return deltaE.mean()

class ReproductionErrorLoss(nn.Module):
    def __init__(self):
        super(ReproductionErrorLoss, self).__init__()
        self.safe_v = 0.999999
    
    def forward(self, img1, img2):
        img1 = img1 / (torch.norm(img1, dim=1, keepdim=True) + 1e-8)
        img2 = img2 / (torch.norm(img2, dim=1, keepdim=True) + 1e-8)

        cos = torch.sum(img1 * img2, dim=1).clamp(-self.safe_v, self.safe_v)
        angle = torch.acos(cos)

        return torch.mean(angle)


# Metrics

class ReproductionError():
    """ 
    Pixelwise angular error between two images
    """
    def __init__(self):
        self.name = "ReproductionError"
        self.safe_v = 0.999999

    def __call__(self, img1, img2):
        b, c, h, w = img1.shape

        img1 = img1.reshape(b, c, -1)
        img2 = img2.reshape(b, c, -1)

        img1 = img1 / (torch.norm(img1, dim=1, keepdim=True) + 1e-8)
        img2 = img2 / (torch.norm(img2, dim=1, keepdim=True) + 1e-8)

        cos = torch.sum(img1 * img2, dim=1).clamp(-self.safe_v, self.safe_v)
        angle = torch.acos(cos) * 180 / math.pi

        angle = angle.cpu()

        return angle.mean(dim=1).tolist()

class AngularError():
    def __init__(self, safe_v = 0.999999):
        super(AngularError, self).__init__()
        self.safe_v = safe_v

    def __call__(self, y_hat, y):
        if len(y_hat.shape) == 4 and len(y.shape) == 2:
            b, c, h, w = y_hat.shape
            y = y[:,:, None, None].expand_as(y_hat)

        y_hat = y_hat / (torch.norm(y_hat, dim=1, keepdim=True) + 1e-8)
        y = y / (torch.norm(y, dim=1, keepdim=True) + 1e-8)

        cos = torch.sum(y_hat * y, dim=1).clamp(-self.safe_v, self.safe_v)
        angle = torch.acos(cos)

        return angle.tolist()


class deltaE76():

    def __init__(self, input_type="xyz"):
        super().__init__()
        self.name = "deltaE76Loss"
    
    def xyz2lab(self, xyz, ill_xyz=torch.Tensor([0.95047, 1., 1.08883])):
            x = xyz[:,0,:,:] 
            y = xyz[:,1,:,:]
            z = xyz[:,2,:,:]

            X_n = ill_xyz[0]
            Y_n = ill_xyz[1]
            Z_n = ill_xyz[2]

            f = lambda t : torch.where(t>(6/29)**3,  t**(1/3), (t*(6/29)**(-2))/3 + (4/29))

            L = 116 * f(y/Y_n) - 16
            a = 500 * (f(x/X_n) - f(y/Y_n))
            b = 200 * (f(y/Y_n) - f(z/Z_n))

            Lab = torch.stack([L,a,b], dim=1)

            return Lab 

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        img1, img2: torch.Tensor of shape BCHW, float32 or float64
        Returns: mean ΔE76 over all pixels and batch
        """
        b, c, h, w = img1.shape
        img1 = self.xyz2lab(img1)
        img2 = self.xyz2lab(img2)

        # Flatten pixels
        arr1 = img1.permute(0, 2, 3, 1).reshape(b, -1, 3)
        arr2 = img2.permute(0, 2, 3, 1).reshape(b, -1, 3)

        Lstd, astd, bstd = arr1[...,0], arr1[...,1], arr1[...,2]
        Lsample, asample, bsample = arr2[...,0], arr2[...,1], arr2[...,2]

        deltaE = torch.sqrt((Lstd - Lsample)**2 + (astd - asample)**2 + (bstd - bsample)**2)

        # Per image average
        avg = deltaE.mean(dim=1)  # (B,) 
        return avg.tolist()

class deltaE00():
    """ΔE00 metric for batch of images (PyTorch only)."""

    def __init__(self, input_type="xyz"):
        self.kl = 1.0
        self.kc = 1.0
        self.kh = 1.0
        self.input_type = input_type
        if input_type != "xyz":
            raise NotImplementedError("Only 'xyz' input is supported in this optimized version.")

    def xyz2lab(self, xyz, ill_xyz=torch.Tensor([0.95047, 1., 1.08883])):
            x = xyz[:,0,:,:] 
            y = xyz[:,1,:,:]
            z = xyz[:,2,:,:]

            X_n = ill_xyz[0]
            Y_n = ill_xyz[1]
            Z_n = ill_xyz[2]

            f = lambda t : torch.where(t>(6/29)**3,  t**(1/3), (t*(6/29)**(-2))/3 + (4/29))

            L = 116 * f(y/Y_n) - 16
            a = 500 * (f(x/X_n) - f(y/Y_n))
            b = 200 * (f(y/Y_n) - f(z/Z_n))

            Lab = torch.stack([L,a,b], dim=1)

            return Lab 

    def convert(self, img: torch.Tensor):
        """
        Convert BCHW XYZ image to LAB (approximate).
        Assumes img is a torch.Tensor on any device.
        """
        # Expect BCHW
        if img.ndim != 4 or img.shape[1] != 3:
            raise ValueError("Input must be BCHW with 3 channels")
        # Convert to Lab
        img = self.xyz2lab(img)

        return img 

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        img1, img2: torch.Tensor of shape BCHW, float32 or float64
        Returns: list of per-image ΔE00 averages
        """
        b, c, h, w = img1.shape
        img1 = self.convert(img1)
        img2 = self.convert(img2)

        # Flatten pixels
        arr1 = img1.permute(0, 2, 3, 1).reshape(b, -1, 3)
        arr2 = img2.permute(0, 2, 3, 1).reshape(b, -1, 3)

        Lstd, astd, bstd = arr1[...,0], arr1[...,1], arr1[...,2]
        Lsample, asample, bsample = arr2[...,0], arr2[...,1], arr2[...,2]

        # Chroma and hue
        Cabstd = torch.sqrt(astd**2 + bstd**2)
        Cabsample = torch.sqrt(asample**2 + bsample**2)
        Cabmean = (Cabstd + Cabsample)/2
        G = 0.5 * (1 - torch.sqrt(Cabmean**7 / (Cabmean**7 + 25**7)))

        apstd = (1 + G) * astd
        apsample = (1 + G) * asample
        Cpstd = torch.sqrt(apstd**2 + bstd**2)
        Cpsample = torch.sqrt(apsample**2 + bsample**2)
        Cpprod = Cpstd * Cpsample

        hpstd = torch.atan2(bstd, apstd)
        hpstd = torch.where((apstd.abs() + bstd.abs())==0, torch.zeros_like(hpstd), hpstd)

        hpsample = torch.atan2(bsample, apsample)
        hpsample = torch.where((apsample.abs() + bsample.abs())==0, torch.zeros_like(hpsample), hpsample)
        hpsample = hpsample + 2*math.pi*(hpsample<0)

        # Differences
        dL = Lsample - Lstd
        dC = Cpsample - Cpstd
        dhp = hpsample - hpstd
        dhp = dhp - 2*math.pi*(dhp > math.pi)
        dhp = dhp + 2*math.pi*(dhp < -math.pi)
        dhp = torch.where(Cpprod==0, torch.zeros_like(dhp), dhp)
        dH = 2 * torch.sqrt(Cpprod) * torch.sin(dhp/2)

        # Averages
        Lp = (Lstd + Lsample)/2
        Cp = (Cpstd + Cpsample)/2
        hp = (hpstd + hpsample)/2
        hp = torch.where(Cpprod==0, hpsample + hpstd, hp)

        # Weighting functions
        Lpm502 = (Lp - 50)**2
        Sl = 1 + 0.015*Lpm502 / torch.sqrt(20 + Lpm502)
        Sc = 1 + 0.045*Cp
        T = 1 - 0.17*torch.cos(hp - math.pi/6) + 0.24*torch.cos(2*hp) \
            + 0.32*torch.cos(3*hp + math.pi/30) - 0.20*torch.cos(4*hp - 63*math.pi/180)
        Sh = 1 + 0.015*Cp*T
        delthetarad = (30*math.pi/180) * torch.exp(-(((180/hp - 275)/25)**2))
        Rc = 2 * torch.sqrt(Cp**7 / (Cp**7 + 25**7))
        RT = -torch.sin(2*delthetarad) * Rc

        # Final ΔE00
        de00 = torch.sqrt(
            (dL/(self.kl*Sl))**2 +
            (dC/(self.kc*Sc))**2 +
            (dH/(self.kh*Sh))**2 +
            RT * (dC/(self.kc*Sc)) * (dH/(self.kh*Sh))
        )

        # Per-image average
        avg = de00.mean(dim=1)  # (B,)
        return avg.tolist()


class LPIPS(nn.Module):
    def __init__(self, net_type="alex"):
        super(LPIPS, self).__init__()
        self.name = "LPIPS"
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=True, reduction="none")

    def forward(self, y_hat, y):
        # return [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        if self.lpips.device != y_hat.device:
            self.lpips = self.lpips.to(y_hat.device)
        return self.lpips(y_hat, y).tolist()


class PSNR():
    def __init__(self, max_val: float = 1.0):
        """
        Peak Signal-to-Noise Ratio (PSNR) metric.

        Args:
            max_val (float): Maximum possible pixel value (1.0 for normalized, 255 for 8-bit)
        """
        self.max_val = max_val

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute PSNR between two images.

        Args:
            img1 (torch.Tensor): First image, shape (C, H, W) or (N, C, H, W)
            img2 (torch.Tensor): Second image, same shape as img1

        Returns:
            torch.Tensor: PSNR value in dB
        """
        img1 = img1.float()
        img2 = img2.float()

        # Compute MSE per image in the batch
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        psnr = 10 * torch.log10((self.max_val ** 2) / mse)
        
        # Handle case where MSE is zero (identical images)
        psnr[mse == 0] = float('inf')
        return psnr.tolist()



