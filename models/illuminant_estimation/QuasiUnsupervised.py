import torch
import torch.nn.functional as F
import gdown
import os

import ipdb

INPUT_SCALED_INT = 1
INPUT_EQUALIZED_INT = 2
INPUT_NORMALIZED_INT = 4
INPUT_NORMALIZED_GRADIENTS_OLD = 8
INPUT_NORMALIZED_GRADIENTS = 16


_INPUT_CHANNELS = {
    INPUT_SCALED_INT: 1,
    INPUT_EQUALIZED_INT: 1,
    INPUT_NORMALIZED_INT: 1,
    INPUT_NORMALIZED_GRADIENTS_OLD: 6,
    INPUT_NORMALIZED_GRADIENTS: 6
}


_INPUT_CODES = {
    "s": INPUT_SCALED_INT,
    "e": INPUT_EQUALIZED_INT,
    "n": INPUT_NORMALIZED_INT,
    "d": INPUT_NORMALIZED_GRADIENTS_OLD,
    "g": INPUT_NORMALIZED_GRADIENTS
}


OUTPUT_RGB = 1
OUTPUT_GRADIENT_MAGNITUDES = 2
OUTPUT_GRADIENT_MAGNITUDES_D4 = 4


_OUTPUT_CODES = {
    "c": OUTPUT_RGB,
    "m": OUTPUT_GRADIENT_MAGNITUDES,
    "M": OUTPUT_GRADIENT_MAGNITUDES_D4,
}


def _input_from_code(code):
    return sum(_INPUT_CODES[c] for c in code)


def _output_from_code(code):
    return sum(_OUTPUT_CODES[c] for c in code)


def compute_estimate(rgb, weights, noise=0.0):
    # rgb =  torch.nn.functional.normalize(rgb)  # !!!
    x = torch.sum(torch.sum(rgb * weights, 3), 2)
    x = x + noise * torch.randn_like(x)
    return torch.nn.functional.normalize(x)


def __compute_estimate(rgb, weights):
    # Basic version
    x = torch.sum(torch.sum(rgb * weights, 3), 2)
    return torch.nn.functional.normalize(x)


def __compute_estimate(rgb, weights):
    # From gradients
    g = spatial_gradient(rgb)
    m = gradient_magnitude(g)
    x = torch.sum(torch.sum(m * weights, 3), 2)
    return torch.nn.functional.normalize(x)


def cosine_loss(estimate, target, reduce=True):
    # Assume all vectors already have unit norm
    cosines = torch.sum(estimate * target, 1)
    loss = 1 - cosines
    angular_error = 180 * torch.acos(torch.clamp(cosines, -1, 1)) / 3.141592653589793
    if reduce:
        loss = torch.mean(loss)
        angular_error = torch.mean(angular_error)
    return loss, angular_error


def apply_correction(rgb, estimates):
    norm = torch.prod(estimates, 1, keepdim=True) ** (1.0 / 3)
    out = norm[:, :, None, None] * rgb / (1e-6 + estimates[:, :, None, None])
    return torch.clamp(out, 0, 1)


def scaled_intensity(rgb):
    y = torch.mean(rgb, 1, keepdim=True)
    ymin = torch.min(y.view(y.size(0), -1), 1)[0]
    y = y - ymin[:, None, None, None]
    ymax = torch.max(y.view(y.size(0), -1), 1)[0]
    y = y / (1e-7 + ymax)[:, None, None, None]
    return y


_sobel_mask = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
_sobel_filters = torch.stack([_sobel_mask, _sobel_mask.t()] * 3, 0).view(-1, 1, 3, 3)


def spatial_gradients(x):
    # input x : BxCxHxW
    # output y: Bx(2C)xHxW
    # the horiz. and vert. components of the gradient of x[b, c, i, j]
    # are out[b, 2*c, i, j] and out[b, 2*c+1, i, j]
    w = _sobel_filters.to(x.device)
    y = torch.nn.functional.conv2d(x, w, padding=1, groups=x.size(1))
    return y


def gradient_magnitude(g):
    # input g  : Bx(2C)xHxW
    # output m : BxCxHxW
    n, _, h, w = g.size()
    return torch.norm(g.view(n, -1, 2, h, w), 2, 2)


def normalize_gradients(g):
    # input g  : Bx(2C)xHxW
    # output m : Bx(2C)xHxW
    n, _, h, w = g.size()
    nrm = torch.nn.functional.normalize(g.view(n, -1, 2, h, w), 2, 2)
    return nrm.view(n, -1, h, w)


def scaled_intensity_and_gradient_directions(rgb):
    # y = scaled_intensity(rgb)
    g = normalize_gradients(spatial_gradients(rgb))
    return torch.cat([y, g], 1)


def _equalize1(image, vmin, vmax, bins):
    scaled = (image - vmin) * (bins - 2) / (vmax - vmin)
    indices = scaled.long()
    hist = torch.bincount(indices.view(-1), minlength=bins)
    chist = torch.cumsum(hist, 0).float()
    chist -= chist[0]
    chist /= chist[-1].clamp(min=1)
    f = scaled - indices.float()
    interpolated = (1 - f) * chist[indices] + f * chist[indices + 1]
    return interpolated
    

def histogram_equalization(data, dim=1, vmin=0, vmax=1, bins=256):
    """Histogram equalization.
    
    Normalize elements in data.  The normalization is performed
    indipendently from elements starting from dimension `dim'
    (e.g. dim=1 normalizes independently each element in a batch).

    vmin, vmax and bins defin the quantization used for the histogram
    computation.

    """
    sz = 1
    for i in range(dim):
        sz *= data.size(i)
    vdata = data.view(sz, -1)
    output = torch.empty_like(vdata)
    for i in range(sz):
        output[i] = _equalize1(vdata[i], vmin, vmax, bins)
    return output.view(data.size())



# https://drive.google.com/file/uc?id=1jUYGjHvi8aNOpymcAoup-dsy2f5nuixM


class BaseCCNet(torch.nn.Module):
    """Network for illuminant estimation.

    Given a RGB input image computes the color of the illuminant.

    """
    def __init__(self, noise=100.0, input_code="sd", output_code="c", mask_clipped=False):
        super().__init__()
        self.noise = noise
        self.mask_clipped = mask_clipped
        self.input_channels = 0
        self.output_channels = len(output_code)
        self.input = _input_from_code(input_code)
        self.output = _output_from_code(output_code)
        for k, v in _INPUT_CHANNELS.items():
            if k & self.input:
                self.input_channels += v

    def preprocessing(self, rgb):
        """Given the image, compute the actual imput to the CNN."""
        data = []
        if self.input & INPUT_SCALED_INT:
            gray = scaled_intensity(rgb)
            data.append(2 * gray - 1)
        if self.input & INPUT_EQUALIZED_INT:
            gray = torch.mean(rgb, 1, keepdim=True)
            gray = histogram_equalization(gray, bins=64)
            data.append(2 * gray - 1)
        if self.input & INPUT_NORMALIZED_INT:
            gray = torch.mean(rgb, 1, keepdim=True)
            gray = F.instance_norm(gray)
            data.append(gray)
        if self.input & INPUT_NORMALIZED_GRADIENTS_OLD:
            gradients = spatial_gradients(rgb)
            directions = normalize_gradients(gradients)
            data.append(2 * directions - 1)
        if self.input & INPUT_NORMALIZED_GRADIENTS:
            gradients = spatial_gradients(rgb)
            directions = normalize_gradients(gradients)
            data.append(directions)
        if len(data) == 1:
            return data[0]
        else:
            return torch.cat(data, 1)

    def compute_rgb_data(self, rgb):
        """Given the image, compute the color data to combine to get the estimate."""
        data = []
        if self.output & OUTPUT_RGB:
            data.append(rgb)
        if self.output & OUTPUT_GRADIENT_MAGNITUDES:
            grad = spatial_gradients(rgb)
            data.append(gradient_magnitude(grad))
        if self.output & OUTPUT_GRADIENT_MAGNITUDES_D4:
            grad = spatial_gradients(rgb)
            data.append(gradient_magnitude(grad) / 4.0)
        if len(data) == 1:
            return data[0].unsqueeze(2)
        else:
            return torch.stack(data, 2)

    def make_estimate(self, rgb, weights, noise=None, mask_clipped=None):
        """Compute the estimate given rgb and weights.

        rgb: Bx3xHxW
        weights: BxCxHxW

        result: Bx3
        """
        if mask_clipped is None:
            mask_clipped = self.mask_clipped
        if mask_clipped:
            ma = rgb.max(1, keepdim=True)[0]
            mm = ma.view(ma.size(0), 1, -1).max(2)[0].unsqueeze(-1).unsqueeze(-1)
            weights = weights * (ma < mm).float()
        data = self.compute_rgb_data(rgb)
        data = data.view(data.size(0), 3, -1)
        weights = weights.view(weights.size(0), 1, -1)
        estimate = torch.sum(data * weights, 2)
        if self.training:
            if noise is None:
                noise = self.noise
            estimate = estimate + noise * torch.randn_like(estimate)
        return torch.nn.functional.normalize(estimate)

    def last_parameters(self):
        """PArameters to be used for transfer learning."""
        return []


def conv_module(in_channels, out_channels, batch_normalization=True, relu=True):
    layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
    if batch_normalization:
        layers.append(torch.nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(torch.nn.LeakyReLU(0.2))
    return torch.nn.Sequential(*layers)


def deconv_module(in_channels, out_channels, dropout=False, batch_normalization=True, relu=True):
    layers = [torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
    if batch_normalization:
        layers.append(torch.nn.BatchNorm2d(out_channels))
    if dropout:
        layers.append(torch.nn.Dropout2d())
    if relu:
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)


def id_layer(x):
    return x


class UNetModule(torch.nn.Module):
    """Compute a BxIxHxW -> BxOxHxW transform.

    The module performs a convolutional block, a deconvolutional block
    and optionally another module in the middle.

    When skip is False the number of output channels O is the same of
    the number of input channels I.  When skip is True O is double of
    I.

    When not None the inner module is expected to keep the image
    dimensions, keeping also constant the number of channels when skip
    is False, or doubling it when skip is True.

    """
    def __init__(self, channels, mid_channels, inner_module=None, skip=True, dropout=False):
        super().__init__()
        self.down = conv_module(channels, mid_channels)
        self.inner = (inner_module if inner_module is not None else id_layer)
        inner_channels = 2 * mid_channels if inner_module is not None and skip else mid_channels
        self.up = deconv_module(inner_channels, channels, dropout=dropout)
        self.skip = skip
        
    def forward(self, x):
        y = self.up(self.inner(self.down(x)))
        if self.skip:
            y = torch.cat([x, y], 1)
        return y

    def decoder_parameters(self):
        p = list(self.up.parameters())
        if self.inner is not id_layer:
            p.extend(self.inner.decoder_parameters())
        return p

    
class QuasiUnsupervised(BaseCCNet):
    """Network for illuminant estimation.

    Given a RGB input image computes the color of the illuminant.

    """
    def __init__(self, skip=True, pretrained=True, **kwargs):
        super().__init__(**kwargs)
        self.conv = conv_module(self.input_channels, 64,
                                batch_normalization=False)
        m = UNetModule(512, 512, None, dropout=True, skip=skip)
        m = UNetModule(512, 512, m, dropout=True, skip=skip)
        m = UNetModule(512, 512, m, dropout=True, skip=skip)
        m = UNetModule(512, 512, m, skip=skip)
        m = UNetModule(256, 512, m, skip=skip)
        m = UNetModule(128, 256, m, skip=skip)
        m = UNetModule(64, 128, m, skip=skip)
        self.inner = m
        channels = (128 if skip else 64)
        self.deconv = deconv_module(channels, self.output_channels,
                                    batch_normalization=False,
                                    relu=False)

        self.model_urls = {
            "ilsvrc12-eg": 'https://drive.google.com/file/uc?id=1jUYGjHvi8aNOpymcAoup-dsy2f5nuixM',
        }

        if pretrained:
            self.load()

    def load(self, model="ilsvrc12-eg"):
        if not os.path.exists(os.path.join("assets", "pretrained", model+".pt")):
            os.makedirs(os.path.join("assets", "pretrained"), exist_ok=True)
            gdown.download(self.model_urls[model], output=os.path.join("assets", "pretrained", model+".pt"), fuzzy=True)
        # ipdb.set_trace()
        self.load_state_dict(torch.load(os.path.join("assets", "pretrained", model+".pt"), weights_only=False)["model"])

    def forward(self, rgb):
        x = self.preprocessing(rgb)
        logits = self.deconv(self.inner(self.conv(x)))
        weights = torch.sigmoid(logits)
        estimate = self.make_estimate(rgb, weights)
        return estimate#, x, weights

    def last_parameters(self):
        # p = self.inner.decoder_parameters()
        # p.extend(self.deconv.parameters())
        # return p
        return self.deconv.parameters()


class SmallCCNet(BaseCCNet):
    """Network for illuminant estimation.

    Given a RGB input image computes the color of the illuminant.

    """
    def __init__(self, skip=True, **kwargs):
        super().__init__(**kwargs)
        self.conv = conv_module(self.input_channels, 32,
                                batch_normalization=False)
        m = UNetModule(256, 256, None, dropout=True, skip=skip)
        m = UNetModule(256, 256, m, skip=skip)
        m = UNetModule(128, 256, m, skip=skip)
        m = UNetModule(64, 128, m, skip=skip)
        m = UNetModule(32, 64, m, skip=skip)
        self.inner = m
        channels = (64 if skip else 32)
        self.deconv = deconv_module(channels, self.output_channels,
                                    batch_normalization=False,
                                    relu=False)

    def forward(self, rgb):
        x = self.preprocessing(rgb)
        logits = self.deconv(self.inner(self.conv(2 * x - 1)))
        weights = torch.sigmoid(logits)
        estimate = self.make_estimate(rgb, weights)
        return estimate, x, weights

    def last_parameters(self):
        return self.deconv.parameters()


# def _test():
#     net = CCNet(input_code="g", output_code="c", mask_clipped=True)
#     print(net)
#     x = torch.rand(2, 3, 256, 256)
#     y = net(x)[0]
#     print(x.size(), "->", y.size())


# if __name__ == "__main__":
#     _test()
