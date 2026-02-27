from .color_reproduction.LPIENet import LPIENet
from .color_reproduction.SpectralLPIENet import SpectralLPIENet
from .color_reproduction.cmKAN import CmKAN, LightCmKAN
from .color_reproduction.SpectralCmKAN import SpectralLightCmKAN
from .illuminant_estimation.FC4 import FC4
from .illuminant_estimation.CCC import CCC
from .illuminant_estimation.ConvMean import ConvMean
from .illuminant_estimation.QuasiUnsupervised import QuasiUnsupervised
from .illuminant_estimation.ConvolutionalEB import ConvolutionalEB
from .color_correction.classic_pipeline import ClassicCorrectionPipeline
from .color_reproduction.MSIAWBNet import MSIAWBNet


# TODO: Implement FFCC

# TODO: Implement MS based methods for comparisons
# TODO: Adapt LPIENet to the task


__all__ = [
    'ConvolutionalEB',
    'ConvMean',
    'QuasiUnsupervised',
    'FC4',
    'CCC',
    "ClassicCorrectionPipeline",
    "MSIAWBNet",
    'LPIENet', 
    "CmKAN",
    "LightCmKAN",
    "SpectralLightCmKAN",
    "SpectralLPIENet"
    ]