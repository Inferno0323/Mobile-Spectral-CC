import os
from typing import Union
import torch
from torch import nn, Tensor
import torch.nn.init as init
from torch.nn.functional import normalize
from torch.utils import model_zoo


class Fire(nn.Module):

    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)),
                          self.expand3x3_activation(self.expand3x3(x))], 1)

class SqueezeNet(nn.Module):

    def __init__(self, version: float = 1.0, num_classes: int = 1000):
        super().__init__()

        self.num_classes = num_classes

        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == 1.1:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError("Unsupported SqueezeNet version {version}: 1.0 or 1.1 expected".format(version=version))

        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)

class SqueezeNetLoader:
    def __init__(self, version: float = 1.1):
        self.__version = version
        self.__model = SqueezeNet(self.__version)

        self.model_urls = {
            1.0: 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
            1.1: 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
        }

    def load(self, pretrained: bool = False) -> SqueezeNet:
        """
        Returns the specified version of SqueezeNet
        @param pretrained: if True, returns a model pre-trained on ImageNet
        """
        if pretrained:
            path_to_local = os.path.join("assets", "pretrained")
            print("\n Loading local model at: {} \n".format(path_to_local))
            os.environ['TORCH_HOME'] = path_to_local
            self.__model.load_state_dict(model_zoo.load_url(self.model_urls[self.__version]))
        return self.__model

class FC4(torch.nn.Module):

    def __init__(self, squeezenet_version: float = 1.1, confidence_weighted_pooling=True, pretrained=True, inp_size=None, resize=True, input_size=None):
        super().__init__()

        # SqueezeNet backbone (conv1-fire8) for extracting semantic features
        squeezenet = SqueezeNetLoader(squeezenet_version).load(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(squeezenet.children())[0][:12])
        if inp_size is not None:
            self.backbone[0] = nn.Conv2d(inp_size, 64, kernel_size=3, stride=2)

        # Final convolutional layers (conv6 and conv7) to extract semi-dense feature maps
        self.final_convs = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True),
            nn.Conv2d(512, 64, kernel_size=6, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 4 if confidence_weighted_pooling else 3, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

        self.confidence_weighted_pooling = confidence_weighted_pooling

        self.resize = resize
        self.input_size = input_size

    def forward(self, x: Tensor) -> Union[tuple, Tensor]:
        """
        Estimate an RGB colour for the illuminant of the input image
        @param x: the image for which the colour of the illuminant has to be estimated
        @return: the colour estimate as a Tensor. If confidence-weighted pooling is used, the per-path colour estimates
        and the confidence weights are returned as well (used for visualizations)
        """

        if self.input_size is not None and x.shape[2:] != (self.input_size, self.input_size):
            x = nn.functional.interpolate(x, size=(self.input_size, self.input_size), mode='bilinear', align_corners=False)
        elif self.resize and (x.shape[2] < 224 or x.shape[3] < 224):
            # if shape is lower than 224x224, resize to 224x224
            x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)


        x = self.backbone(x)
        out = self.final_convs(x)
        # Confidence-weighted pooling: "out" is a set of semi-dense feature maps
        if self.confidence_weighted_pooling:
            # Per-patch color estimates (first 3 dimensions)
            rgb = normalize(out[:, :3, :, :], dim=1)

            # Confidence (last dimension)
            confidence = out[:, 3:4, :, :]

            out = torch.sum(torch.sum(rgb * confidence, 2), 2)
            
            # Confidence-weighted pooling
            pred = normalize(out+1e-9, dim=1)

            return pred#, rgb, confidence
        
        # Summation pooling
        out = torch.sum(torch.sum(out, 2), 2)
        pred = normalize(out+1e-9, dim=1)

        return pred