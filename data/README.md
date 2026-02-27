# Dataset Structure and Generation

## Dataset Structure

```text
GT
├── xyz_scenes        # Ground-truth scenes rendered under D65 with CIE XYZ 1931 sensitivities (12-bit)
├── srgb_scenes       # Ground-truth scenes rendered under D65 in sRGB (8-bit)
└── illums_spd        # Illuminant spectral power distributions (SPD), stored in .h5

<MS_CAMERA_NAME>      # Example: SPECTRICITY_S1 (multispectral camera)
├── imgs              # Rendered scenes (bit depth depends on camera, e.g., 10-bit)
├── illums_spd        # Illuminant SPDs measured with the camera sensitivities (.h5)
└── gt_raw            # Ground-truth raw images rendered with camera sensitivities under D65

<RGB_CAMERA_NAME>     # Example: CANON_R5 (RGB camera)
├── imgs              # Captured scenes (bit depth depends on camera, e.g., 12-bit)
├── gt_raw            # Ground-truth raw images rendered with camera sensitivities under D65
└── metadata          # Illuminant-specific metadata (.json)
```

> **Note:**  
> `SPECTRICITY_S1` and `CANON_R5` are provided as examples of a multispectral (MS) camera and an RGB camera, respectively. The same folder structure applies to any supported MS or RGB camera model included in the dataset.

---

## Spectral Image Format

Spectral images are stored in `.h5` format as 3D arrays with shape **(C, H, W)**:

- **`wvs`** — 1D array of wavelengths (one per channel)  
- **`data`** — 3D array containing spectral data, where each channel corresponds to a wavelength  

---

## Metadata Format

Metadata files are stored in `.json` format. Each entry corresponds to a specific illuminant condition and includes:

- **`illuminant_rgb`** — Normalized RGB values of the illuminant  
- **`ill_cct`** — Matrix converting illuminant raw RGB to XYZ  
- **`cct_base`** — Matrix converting white-balanced raw RGB to XYZ under any illuminant  
- **`cct_2500K`** — Matrix optimized for white-balanced raw RGB to XYZ under 2500K  
- **`cct_6500K`** — Matrix optimized for white-balanced raw RGB to XYZ under 6500K  
- **`cct_interp`** — Matrix converting white-balanced raw RGB to XYZ under the given illuminant, obtained by interpolating between `cct_2500K` and `cct_6500K` based on the illuminant’s correlated color temperature (CCT)  

---

## Dataset Generation

You can download the dataset from:  
https://huggingface.co/datasets/LucaCogo/MobileSpectralCCDataset  

Place it in:

```bash
data/MobileSpectralCCDataset/
```

### Generate the Dataset Manually

Alternatively, generate the dataset using:

```bash
scripts/generate_dataset.py
```

Download the required reflectance datasets (KAUST and BJTU-UVA), and organize them as follows:

```bash
data/KAUST_AWB
data/BJTU_UVA/31bands_h5
```