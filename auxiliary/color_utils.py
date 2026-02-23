import numpy as np
from colour import SDS_ILLUMINANTS, MSDS_CMFS, SpragueInterpolator, SpectralShape, SDS_COLOURCHECKERS
from colour.colorimetry import blackbody_spectral_radiance
import torch
import math
import sys
import os

from auxiliary.io_utils import read_ssf, read_reflectance_h5, get_ssf_path_for_camera, read_homographies, scale_homography, warp_and_crop, HOMOGRAPHY_REF_SIZE

class Illuminant():
        def __init__(self, ill_name=None, spd=None, wv=None):
            assert ((ill_name is not None) ^ (spd is not None and wv is not None)), "Only one between (ill_name) and (spd, wv) should be specified"

            if spd is not None:
                self.spd = spd
                self.wv = wv
            
            if ill_name is not None:
                self.spd =  SDS_ILLUMINANTS[ill_name].values
                self.wv = SDS_ILLUMINANTS[ill_name].wavelengths

        def get_spd(self, new_wv=None):
            if np.all(new_wv == None) or (new_wv.shape == self.wv.shape and np.all(new_wv == self.wv)):
                return self.spd

            sprague_interpolator = SpragueInterpolator(self.wv, self.spd)
            spd = sprague_interpolator(new_wv)

            return spd
        
class CMF():
        def __init__(self, cmf_name="CIE 1931 2 Degree Standard Observer"):
            self.cmf = MSDS_CMFS[cmf_name].values.T
            self.wv = MSDS_CMFS[cmf_name].wavelengths

        def get_cmf(self, new_wv=None):
            if np.all(new_wv == None) or (new_wv.shape == self.wv.shape and np.all(new_wv == self.wv)):
                return self.cmf
            new_cmf = np.array([SpragueInterpolator(self.wv, self.cmf[i])(new_wv) for i in range(3)])

            return new_cmf
        
def spec2xyz(spec, ill, cmf):
        b,c,h,w = spec.shape
        
        x = spec * ill[None,:,None,None]
        xyz = cmf @ x.permute(1,0,2,3).reshape(c,-1)
        xyz = xyz.reshape(3,b,h,w).permute(1,0,2,3)
        
        ill_y = (ill @ cmf.T)[1]
        
        xyz = xyz/ill_y

        return xyz

def xyz2srgb(xyz):
        b,_,h,w = xyz.shape
        M = torch.tensor([[3.2406, -1.5372, -0.4986],[-0.9689, 1.8758, 0.0414],[0.0557, -0.2040, 1.0570]])

        rgb_lin = M @ xyz.permute(1,0,2,3).reshape(3,-1) 
        rgb_lin = rgb_lin.reshape(3,b,h,w).permute(1,0,2,3)
        rgb_lin_safe = rgb_lin.clamp(min=0.0)

        srgb = torch.where(rgb_lin_safe<=0.0031308, 12.92*rgb_lin_safe, 1.055*(rgb_lin_safe**(1/2.4)) - 0.055)
        srgb = srgb.clamp(0,1)
        
        return srgb

def spec2srgb(spec, ill, cmf):
     return xyz2srgb(spec2xyz(spec,ill,cmf))


class UVT:
    """Helper class representing u, v, t values."""
    def __init__(self, u, v, t):
        self.u = u
        self.v = v
        self.t = t


class CCT_Robertson:
    """
    Correlated Color Temperature (CCT) calculator using Robertson's method.
    Based on XYZ to CCT conversion.
    """

    def __init__(self):
        # Reciprocal temperature (K^-1)
        self.rt = [
            sys.float_info.min, 10.0e-6, 20.0e-6, 30.0e-6, 40.0e-6, 50.0e-6,
            60.0e-6, 70.0e-6, 80.0e-6, 90.0e-6, 100.0e-6, 125.0e-6,
            150.0e-6, 175.0e-6, 200.0e-6, 225.0e-6, 250.0e-6, 275.0e-6,
            300.0e-6, 325.0e-6, 350.0e-6, 375.0e-6, 400.0e-6, 425.0e-6,
            450.0e-6, 475.0e-6, 500.0e-6, 525.0e-6, 550.0e-6, 575.0e-6,
            600.0e-6
        ]

        # UVT reference data
        self.uvt = [
            UVT(0.18006, 0.26352, -0.24341),
            UVT(0.18066, 0.26589, -0.25479),
            UVT(0.18133, 0.26846, -0.26876),
            UVT(0.18208, 0.27119, -0.28539),
            UVT(0.18293, 0.27407, -0.30470),
            UVT(0.18388, 0.27709, -0.32675),
            UVT(0.18494, 0.28021, -0.35156),
            UVT(0.18611, 0.28342, -0.37915),
            UVT(0.18740, 0.28668, -0.40955),
            UVT(0.18880, 0.28997, -0.44278),
            UVT(0.19032, 0.29326, -0.47888),
            UVT(0.19462, 0.30141, -0.58204),
            UVT(0.19962, 0.30921, -0.70471),
            UVT(0.20525, 0.31647, -0.84901),
            UVT(0.21142, 0.32312, -1.0182),
            UVT(0.21807, 0.32909, -1.2168),
            UVT(0.22511, 0.33439, -1.4512),
            UVT(0.23247, 0.33904, -1.7298),
            UVT(0.24010, 0.34308, -2.0637),
            UVT(0.24792, 0.34655, -2.4681),  # corrected
            UVT(0.25591, 0.34951, -2.9641),
            UVT(0.26400, 0.35200, -3.5814),
            UVT(0.27218, 0.35407, -4.3633),
            UVT(0.28039, 0.35577, -5.3762),
            UVT(0.28863, 0.35714, -6.7262),
            UVT(0.29685, 0.35823, -8.5955),
            UVT(0.30505, 0.35907, -11.324),
            UVT(0.31320, 0.35968, -15.628),
            UVT(0.32129, 0.36011, -23.325),
            UVT(0.32931, 0.36038, -40.770),
            UVT(0.33724, 0.36051, -116.45),
        ]

    @staticmethod
    def lerp(a, b, c):
        """Linear interpolation."""
        return ((b - a) * c + a)

    def XYZtoCorColorTemp(self, xyz):
        """
        Convert CIE XYZ to correlated color temperature (Kelvin).
        Args:
            xyz: iterable of (X, Y, Z)
        Returns:
            (status, temp)
              status = 0 if success, -1 if failure
              temp   = temperature in Kelvin if success, None if fail
        """
        X, Y, Z = xyz

        if (X < 1.0e-20) and (Y < 1.0e-20) and (Z < 1.0e-20):
            return -1, None  # protect against divide-by-zero failure

        us = (4.0 * X) / (X + 15.0 * Y + 3.0 * Z)
        vs = (6.0 * Y) / (X + 15.0 * Y + 3.0 * Z)

        dm = 0.0
        for i in range(len(self.uvt)):
            di = (vs - self.uvt[i].v) - self.uvt[i].t * (us - self.uvt[i].u)
            if i > 0 and ((di < 0.0 <= dm) or (di >= 0.0 > dm)):
                break  # found bounding lines
            dm = di
        else:
            return -1, None  # temp below 1666.7K or too blue

        di /= math.sqrt(1.0 + self.uvt[i].t ** 2)
        dm /= math.sqrt(1.0 + self.uvt[i - 1].t ** 2)
        p = dm / (dm - di)  # interpolation parameter
        temp = 1.0 / self.lerp(self.rt[i - 1], self.rt[i], p)

        return 0, temp


def CCT_McCamy(xyzs):
    """
    Correlated Color Temperature (CCT) calculator using McCamy's method.
    Based on xy to CCT conversion.

    Args:
        xyzs: Nx3 array of CIE XYZ values

    n = (x-0.3320)/(0.1858-y);
    CCT = 437*n^3 + 3601*n^2 + 6861*n + 5517
    """

    Xs, Ys, Zs = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]

    xs = Xs / (Xs + Ys + Zs)
    ys = Ys / (Xs + Ys + Zs)

    ns = (xs - 0.3320) / (0.1858 - ys)

    ccts = 437 * ns**3 + 3601 * ns**2 + 6861 * ns + 5517

    status = np.where((ccts < 1667) | (ccts > 25000), -1, 0)  # McCamy's formula is valid only in this range
    # ccts = np.where((ccts < 1667) | (ccts > 25000), None, ccts) # McCamy's formula is valid only in this range

    return status, ccts

    
def interpolate_ccms(ill_rgbs, cct1, cct2, ill_cct, temp1=2500, temp2=6500):

    # Correlated color temperature of the illuminant
    xyzs = (ill_cct @ ill_rgbs.T).T

    status, ccts = CCT_McCamy(xyzs)
    if np.any(status != 0):
        print("Warning: CCT calculation failed for some samples!")
        
    gs = (ccts**(-1) - temp2**(-1)) / (temp1**(-1) - temp2**(-1))
    gs = gs[:,None,None]

    ccms = gs * cct1[None,:,:] + (1-gs) * cct2[None,:,:]

    return ccms


# =============================================================================
# On-the-fly image rendering utilities
# =============================================================================

# Standard wavelength range for rendering
TARGET_WVS = np.arange(400, 701, 10)

# Standard illuminants at target wavelengths
D65_SPECTRUM = SDS_ILLUMINANTS['D65'].align(SpectralShape(400, 700, 10)).values
E_SPECTRUM = SDS_ILLUMINANTS['E'].align(SpectralShape(400, 700, 10)).values
XYZ_CMF_SPECTRUM = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].align(SpectralShape(400, 700, 10)).values


def interpolate_spectrum(spec, old_wvs, new_wvs):
    """Interpolate spectrum to new wavelengths using Sprague interpolation."""
    sprague_interpolator = SpragueInterpolator(old_wvs, spec)
    return sprague_interpolator(new_wvs)

def normalize_spectrum(spec):
    """Normalize spectrum to have max value of 1."""
    return spec / np.max(spec)

def render_image_from_refl(refl, ill, ssf):
    """
    Render an image from reflectance data under a given illuminant using a spectral sensitivity function.
    
    Args:
        refl: Reflectance data (C, H, W) where C is the number of spectral bands
        ill: Illuminant SPD (C,)
        ssf: Spectral sensitivity function (C, N) where N is the number of output channels
        
    Returns:
        img: Rendered image (H, W, N) normalized
        ill_response: Illuminant response in camera space (N,) normalized
    """
    c, h, w = refl.shape

    rad = refl * ill[:, None, None]
    
    rad = rad.reshape(c, h*w)
    img = ssf.T @ rad
    img = img.reshape(ssf.shape[1], h, w).transpose(1, 2, 0)
    
    ill_response = ssf.T @ ill
    
    img = img / ill_response.max()
    img = np.clip(img, 0, 1)

    return img, ill_response / np.linalg.norm(ill_response)


def compute_correction_matrix(ssf_rgb, temp=None, ill_name=None):
    """
    Compute color correction matrix from camera RGB to XYZ.
    
    Args:
        ssf_rgb: RGB spectral sensitivity function (C, 3)
        temp: Black body temperature in Kelvin
        ill_name: Illuminant name (e.g., 'D65', 'E')
    
    Returns:
        correction_matrix: 3x3 correction matrix
        integrated_cct: Integrated CCT correction matrix
    """
    assert (temp is not None) ^ (ill_name is not None), "Only one between (temp) and (ill_name) should be specified"
    
    if ill_name is not None:
        bb_spectrum = SDS_ILLUMINANTS[ill_name].align(SpectralShape(400, 700, 10)).values
    else:
        bb_spectrum = np.array([blackbody_spectral_radiance(wv, temp) for wv in np.arange(400, 701, 10)*1e-9])
    
    # Macbeth Color Checker reflectance spectra
    macbeth_spectra = np.array([SDS_COLOURCHECKERS['cc_ohta'][k].align(SpectralShape(400, 700, 10)).values 
                                for k in SDS_COLOURCHECKERS['cc_ohta'].keys()])
    
    # Compute the RGB values for the Macbeth Color Checker under the black body illuminant
    macbeth_rgb = (ssf_rgb.T @ (macbeth_spectra.T * bb_spectrum[:, None]))
    illum = (ssf_rgb.T @ bb_spectrum)
    macbeth_rgb = macbeth_rgb / illum.max()
    macbeth_rgb = (macbeth_rgb * (2**12 - 1)).round() / (2**12 - 1)
    illum = ((illum / illum.max()) * (2**12 - 1)).round() / (2**12 - 1)
    macbeth_rgb = macbeth_rgb / (illum / np.linalg.norm(illum))[:, None]

    # Compute the XYZ values for the Macbeth Color Checker under D65
    macbeth_xyz = (XYZ_CMF_SPECTRUM.T @ (macbeth_spectra.T * D65_SPECTRUM[:, None]))
    d65_max = (XYZ_CMF_SPECTRUM.T @ D65_SPECTRUM).max()
    macbeth_xyz = macbeth_xyz / d65_max
    macbeth_xyz = (macbeth_xyz * (2**12 - 1)).round() / (2**12 - 1)
    
    # Compute the correction matrix from RGB to XYZ
    correction_matrix, _, _, _ = np.linalg.lstsq(macbeth_rgb.T, macbeth_xyz.T, rcond=None)
    integrated_cct = np.diag((illum / np.linalg.norm(illum))**(-1)) @ correction_matrix
    
    return correction_matrix, integrated_cct


def compute_illuminant_correction_matrix(ssf_rgb):
    """
    Compute illuminant correction matrix for CCT interpolation.
    
    Args:
        ssf_rgb: RGB spectral sensitivity function (C, 3)
    
    Returns:
        correction_matrix: 3x3 correction matrix for illuminant RGB to XYZ
    """
    ill_list = []
    xyz_ill_list = []
    
    for temp in np.arange(2500, 6501, 500):
        bb_spectrum = np.array([blackbody_spectral_radiance(wv, temp) for wv in np.arange(400, 701, 10)*1e-9])
        illum = (ssf_rgb.T @ bb_spectrum)
        illum = illum / np.linalg.norm(illum)

        illum_xyz = (XYZ_CMF_SPECTRUM.T @ bb_spectrum)
        illum_xyz = illum_xyz / np.linalg.norm(illum_xyz)

        ill_list.append(illum)
        xyz_ill_list.append(illum_xyz)

    ill_list = np.stack(ill_list, axis=0)
    xyz_ill_list = np.stack(xyz_ill_list, axis=0)

    correction_matrix, _, _, _ = np.linalg.lstsq(ill_list, xyz_ill_list, rcond=None)

    return correction_matrix


def interpolate_single_ccm(ill_rgb, cct1, cct2, ill_cct, temp1=2500, temp2=6500):
    """
    Interpolate color correction matrix based on illuminant CCT.
    
    Args:
        ill_rgb: Illuminant RGB response (3,)
        cct1: CCM at temp1
        cct2: CCM at temp2
        ill_cct: Illuminant correction matrix for CCT computation
        temp1: Lower temperature bound (default 2500K)
        temp2: Upper temperature bound (default 6500K)
    
    Returns:
        ccm: Interpolated color correction matrix
    """
    xyz = ill_rgb @ ill_cct

    status, cct = CCT_Robertson().XYZtoCorColorTemp(xyz)
    if status != 0:
        print("Warning: CCT calculation failed!")
        cct = (temp1 + temp2) / 2  # Default to midpoint
        
    g = (cct**(-1) - temp2**(-1)) / (temp1**(-1) - temp2**(-1))
    g = np.clip(g, 0, 1)  # Ensure interpolation is bounded

    ccm = g * cct1 + (1-g) * cct2

    return ccm


class SpectralImageRenderer:
    """
    Class to handle on-the-fly rendering of spectral images.
    Caches SSF, illuminants, and scene mappings for efficiency.
    """
    
    def __init__(self, 
                 ssf_path=None,
                 spectral_datasets=None,
                 illuminants_path=None,
                 target_wvs=None,
                 scene_mapping=None,
                 bit_depth=12,
                 misaligned=False,
                 homographies_path=None):
        """
        Initialize the renderer.
        
        Args:
            ssf_path: Path to the SSF h5 file
            spectral_datasets: Dict mapping dataset codes to paths, e.g., {"K": "path/to/KAUST", "B": "path/to/BJTU"}
            illuminants_path: Path to illuminants .npy file
            target_wvs: Target wavelengths (default: 400-700nm with 10nm step)
            scene_mapping: Optional pre-computed scene mapping dict
            bit_depth: Bit depth for quantization (default: 12)
            misaligned: Whether to apply misalignment to MS images (default: False)
            homographies_path: Path to directory containing homography .npy files
        """
        self.target_wvs = target_wvs if target_wvs is not None else TARGET_WVS
        self.bit_depth = bit_depth
        self.ssf = None
        self.ssf_interpolated = None
        self.camera_name = None
        self.illuminants = None
        self.illuminants_interpolated = None
        self.spectral_datasets = spectral_datasets or {}
        self.scene_mapping = scene_mapping or {}
        
        # Misalignment support
        self.misaligned = misaligned
        self.homographies = None
        if misaligned:
            homographies_path = homographies_path or "data/Zurich_homographies(512x512)"
            self.homographies = read_homographies(homographies_path)
            print(f"[SpectralImageRenderer] Loaded {len(self.homographies)} homographies for misalignment")
        
        # Cache for correction matrices
        self._ill_cct = None
        self._cct_base = None
        self._cct_2500 = None
        self._cct_6500 = None
        
        # Cache for reflectance data
        self._refl_cache = {}
        
        if ssf_path is not None:
            self.load_ssf(ssf_path)
        
        if illuminants_path is not None:
            self.load_illuminants(illuminants_path)
    
    def load_ssf(self, ssf_path):
        """Load and interpolate SSF to target wavelengths."""
        self.camera_name, ssf_wvs, ssf = read_ssf(ssf_path)
        n_channels = ssf.shape[0]
        self.ssf_interpolated = np.stack(
            [interpolate_spectrum(ssf[i, :], ssf_wvs, self.target_wvs) for i in range(n_channels)], 
            axis=1
        )
        self.ssf = ssf
        
        # Precompute correction matrices for RGB cameras (3 channels)
        if n_channels == 3:
            self._ill_cct = compute_illuminant_correction_matrix(self.ssf_interpolated)
            _, self._cct_base = compute_correction_matrix(self.ssf_interpolated, ill_name="E")
            self._cct_2500, _ = compute_correction_matrix(self.ssf_interpolated, temp=2500)
            self._cct_6500, _ = compute_correction_matrix(self.ssf_interpolated, temp=6500)
    
    def load_illuminants(self, illuminants_path, illuminants_wvs=None):
        """Load and interpolate illuminants to target wavelengths."""
        self.illuminants = np.load(illuminants_path)
        if illuminants_wvs is None:
            illuminants_wvs = np.arange(380, 781, 4)  # SFU dataset wavelengths
        
        self.illuminants_interpolated = np.stack(
            [interpolate_spectrum(ill, illuminants_wvs, self.target_wvs) for ill in self.illuminants],
            axis=0
        )
    
    def get_num_illuminants(self):
        """Return the number of available illuminants."""
        if self.illuminants_interpolated is None:
            return 0
        return len(self.illuminants_interpolated)
    
    def get_reflectance(self, scene_name, dataset_code):
        """
        Get reflectance data for a scene, with caching.
        
        Args:
            scene_name: Scene name (e.g., "SC001")
            dataset_code: Dataset code ("K" for KAUST, "B" for BJTU)
        
        Returns:
            refl: Reflectance data (C, H, W)
        """
        cache_key = f"{scene_name}_{dataset_code}"
        if cache_key not in self._refl_cache:
            if dataset_code not in self.spectral_datasets:
                raise ValueError(f"Unknown dataset code: {dataset_code}")
            
            # Find the scene file
            dataset_path = self.spectral_datasets[dataset_code]
            scene_files = sorted(os.listdir(dataset_path))
            
            # Use scene mapping if available
            if cache_key in self.scene_mapping:
                scene_file = self.scene_mapping[cache_key]
            else:
                # Extract scene index from scene_name (e.g., "SC001" -> 1)
                scene_idx = int(scene_name[2:])
                if scene_idx < len(scene_files):
                    scene_file = scene_files[scene_idx]
                else:
                    raise ValueError(f"Scene index {scene_idx} out of range for dataset {dataset_code}")
            
            refl, _ = read_reflectance_h5(os.path.join(dataset_path, scene_file))
            self._refl_cache[cache_key] = refl
        
        return self._refl_cache[cache_key]
    
    def render_rgb(self, scene_name, dataset_code, illuminant_idx):
        """
        Render an RGB image from spectral data.
        
        Args:
            scene_name: Scene name (e.g., "SC001")
            dataset_code: Dataset code ("K" or "B")
            illuminant_idx: Index of illuminant to use
        
        Returns:
            rgb_image: Rendered RGB image (3, H, W) in CHW format
            metadata: Dict with illuminant info and correction matrices
        """
        if self.ssf_interpolated is None:
            raise ValueError("SSF not loaded. Call load_ssf() first.")
        if self.illuminants_interpolated is None:
            raise ValueError("Illuminants not loaded. Call load_illuminants() first.")
        
        refl = self.get_reflectance(scene_name, dataset_code)
        ill = self.illuminants_interpolated[illuminant_idx]
        
        # Render image
        raw_img, ill_rgb = render_image_from_refl(refl, ill, self.ssf_interpolated)
        
        # Quantize to bit depth
        raw_img = (raw_img * (2**self.bit_depth - 1)).round() / (2**self.bit_depth - 1)
        
        # Compute interpolated CCM
        cct_interp = interpolate_single_ccm(ill_rgb, self._cct_2500, self._cct_6500, self._ill_cct)
        
        metadata = {
            "illuminant_rgb": ill_rgb.astype(np.float32),
            "ill_cct": self._ill_cct.astype(np.float32),
            "cct_base": self._cct_base.astype(np.float32),
            "cct_2500K": self._cct_2500.astype(np.float32),
            "cct_6500K": self._cct_6500.astype(np.float32),
            "cct_interp": cct_interp.astype(np.float32),
        }
        
        # Convert to CHW format
        rgb_image = raw_img.transpose(2, 0, 1).astype(np.float32)
        
        return rgb_image, metadata
    
    def render_gt_rgb(self, scene_name, dataset_code):
        """
        Render ground truth RGB image (under illuminant E).
        
        Args:
            scene_name: Scene name (e.g., "SC001")
            dataset_code: Dataset code ("K" or "B")
        
        Returns:
            gt_image: Ground truth RGB image (3, H, W) in CHW format
        """
        if self.ssf_interpolated is None:
            raise ValueError("SSF not loaded. Call load_ssf() first.")
        
        refl = self.get_reflectance(scene_name, dataset_code)
        
        # Render under illuminant E
        raw_gt, _ = render_image_from_refl(refl, E_SPECTRUM, self.ssf_interpolated)
        
        # Quantize to bit depth
        raw_gt = (raw_gt * (2**self.bit_depth - 1)).round() / (2**self.bit_depth - 1)
        
        # Convert to CHW format
        gt_image = raw_gt.transpose(2, 0, 1).astype(np.float32)
        
        return gt_image
    
    def render_ms(self, scene_name, dataset_code, illuminant_idx, homography_idx=None):
        """
        Render a multispectral image from spectral data.
        
        Args:
            scene_name: Scene name (e.g., "SC001")
            dataset_code: Dataset code ("K" or "B")
            illuminant_idx: Index of illuminant to use
            homography_idx: Index of homography to use for misalignment (None for random)
        
        Returns:
            ms_image: Rendered MS image (C, H, W) in CHW format
            ms_wvs: Wavelengths for each channel
            ms_illum_spd: Illuminant SPD in MS camera space
        """
        if self.ssf_interpolated is None:
            raise ValueError("SSF not loaded. Call load_ssf() first.")
        if self.illuminants_interpolated is None:
            raise ValueError("Illuminants not loaded. Call load_illuminants() first.")
        
        refl = self.get_reflectance(scene_name, dataset_code)
        
        # Apply misalignment if enabled
        if self.misaligned and self.homographies is not None:
            refl = self._apply_misalignment(refl, homography_idx)
        
        ill = self.illuminants_interpolated[illuminant_idx]
        
        # Render image
        raw_img, ill_spd = render_image_from_refl(refl, ill, self.ssf_interpolated)
        
        # Quantize to 10 bits for MS
        raw_img = (raw_img * (2**10 - 1)).astype(np.uint16) / (2**10 - 1)
        ill_spd = (ill_spd * (2**10 - 1)).astype(np.uint16) / (2**10 - 1)
        
        # Compute wavelengths for each MS channel
        ms_wvs = np.array([self.target_wvs[np.argmax(self.ssf_interpolated[:, i])] 
                          for i in range(self.ssf_interpolated.shape[1])])
        
        # Convert to CHW format
        ms_image = raw_img.transpose(2, 0, 1).astype(np.float32)
        
        return ms_image, ms_wvs, ill_spd.astype(np.float32)
    
    def render_gt_ms(self, scene_name, dataset_code, homography_idx=None):
        """
        Render ground truth MS image (under illuminant E).
        
        Args:
            scene_name: Scene name (e.g., "SC001")
            dataset_code: Dataset code ("K" or "B")
            homography_idx: Index of homography to use for misalignment (None for random)
        
        Returns:
            gt_ms: Ground truth MS image (C, H, W) in CHW format
            ms_wvs: Wavelengths for each channel
        """
        if self.ssf_interpolated is None:
            raise ValueError("SSF not loaded. Call load_ssf() first.")
        
        refl = self.get_reflectance(scene_name, dataset_code)
        
        # Apply misalignment if enabled
        if self.misaligned and self.homographies is not None:
            refl = self._apply_misalignment(refl, homography_idx)
        
        # Render under illuminant E
        raw_gt, _ = render_image_from_refl(refl, E_SPECTRUM, self.ssf_interpolated)
        
        # Compute wavelengths for each MS channel
        ms_wvs = np.array([self.target_wvs[np.argmax(self.ssf_interpolated[:, i])] 
                          for i in range(self.ssf_interpolated.shape[1])])
        
        # Convert to CHW format
        gt_ms = raw_gt.transpose(2, 0, 1).astype(np.float32)
        
        return gt_ms, ms_wvs
    
    def _apply_misalignment(self, refl, homography_idx=None):
        """
        Apply misalignment to reflectance data using a homography.
        
        Args:
            refl: Reflectance data (C, H, W)
            homography_idx: Index of homography to use (None for random)
        
        Returns:
            refl_warped: Warped and cropped reflectance data (C, H', W')
        """
        if self.homographies is None:
            return refl
        
        # Select homography
        if homography_idx is None:
            homography_idx = np.random.randint(0, len(self.homographies))
        
        M = self.homographies[homography_idx]
        
        # Scale homography if image size differs from reference
        c, h, w = refl.shape
        if h != HOMOGRAPHY_REF_SIZE or w != HOMOGRAPHY_REF_SIZE:
            M = scale_homography(M, (h, w), HOMOGRAPHY_REF_SIZE)
        
        # Apply warp and crop
        refl_warped = warp_and_crop(refl, M)
        
        return refl_warped
    
    def get_num_homographies(self):
        """Return the number of available homographies."""
        if self.homographies is None:
            return 0
        return len(self.homographies)
    
    def clear_cache(self):
        """Clear the reflectance cache to free memory."""
        self._refl_cache.clear()

