# Configuration Files

This folder contains configuration files for training and testing models on the Mobile Spectral AWB dataset. Each configuration file defines the experiment settings, model architecture, dataset parameters, and training hyperparameters.

## Usage

To run an experiment using a configuration file:

```bash
python run.py conf/<config_file>.py
```

## Configuration Structure

All configuration files follow a consistent structure with a `cfg` dictionary containing the following key sections:

### General Settings

| Parameter | Type | Description |
|-----------|------|-------------|
| `exp_name` | str | Name of the experiment (used for logging and checkpoints) |
| `train` | bool | Set `True` for training, `False` for testing only |
| `test` | bool | Set `True` to run evaluation after training or during inference |

### Model Settings

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_type` | str | Type of model: `"IE"` (Illuminant Estimation), `"MSIE"` (Multispectral IE), or `"J_MSI"` (Joint Multispectral Image-to-Image) |
| `model_name` | str | Name of the model architecture |
| `model_parameters` | dict | Model-specific parameters |
| `input_size` | int | Optional FC4 model parameter to downscale inputs before the backbone (for example `224`) |

### Dataset Settings

| Parameter | Type | Description |
|-----------|------|-------------|
| `data_type` | str | Input data type: `"RGB"`, `"MS"`, or `"RGB+MS"` |
| `dataset_root` | str | Path to the dataset root directory |
| `rgb_camera` | str | RGB camera sensitivity to use (e.g., `"CANON_R5"`) |
| `gt_type` | str | Ground truth type: `"xyz"`, `"srgb"`, or `"raw"` |
| `spectral_camera` | str | Multispectral camera (e.g., `"SPECTRICITY_S1"`, `"SPECTRICITY_S1_MISALIGNED"`) |
| `train_list` / `val_list` / `test_list` | str | Paths to data split files |

### Training Settings

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | int | Random seed for reproducibility |
| `device` | int/list/str | GPU device ID (`-1` for CPU), list of GPU IDs for multi-GPU (for example `[0, 1]`), or `"all"` for all visible GPUs |
| `device_ids` | list | Optional explicit CUDA device order for `DataParallel`; the first ID is the primary device |
| `data_parallel` | bool | Optional override for `DataParallel` usage. By default, single-GPU configs use all visible CUDA devices when more than one GPU is available |
| `n_epochs` | int | Maximum number of training epochs |
| `n_workers` | int | Number of data loading workers |
| `lr` | float | Learning rate |
| `train_batch_size` / `val_batch_size` / `test_batch_size` | int | Batch sizes |
| `early_stop` | int | Early stopping patience (epochs) |
| `criterion` | str | Loss function: `"AngularErrorLoss"`, `"L1Loss"`, `"L2Loss"`, `"deltaE76Loss"` |
| `metrics` | list | Evaluation metrics: `["ReproductionError", "deltaE00", "PSNR", "LPIPS"]` |
| `deterministic` | bool | Keep deterministic CUDA behavior (`True`, default) or allow faster cuDNN autotuned kernels (`False`) |
| `amp` | bool | Enable automatic mixed precision on CUDA for faster training and lower memory use |
| `amp_dtype` | str | AMP dtype: `"float16"` or `"bfloat16"` |
| `tf32` | bool | Enable TF32 matmul/cuDNN kernels on supported NVIDIA GPUs |
| `channels_last` | bool | Use channels-last memory format for CUDA convolution throughput |
| `non_blocking` | bool | Use non-blocking host-to-GPU tensor transfers when pinned memory is enabled |
| `persistent_workers` | bool | Keep DataLoader workers alive across epochs |
| `prefetch_factor` | int | Number of batches each DataLoader worker prefetches |
| `profile_model` | bool | Run startup FLOPs/timing profiling (`True`, default); disable for faster startup |
| `train_metrics` | bool | Compute full training image metrics every batch (`True`, default); set `False` for loss-only training logs and faster FC4/IE training |
| `val_metrics` | bool | Compute full validation image metrics (`True`, default); set `False` for loss-only validation during fast training |
| `val_interval` | int | Run validation every N epochs |
| `cache_rgb` | bool | Cache resized RGB training tensors for fast FC4/RGB training |
| `cache_dir` | str | Optional directory for the fast RGB tensor cache |

### Checkpoints & Pretrained Weights

| Parameter | Type | Description |
|-----------|------|-------------|
| `exp_dir` | str | Directory for experiment outputs (auto-set if `None`) |
| `train_checkpoint` | str | Path to checkpoint to resume training |
| `pretrained_weights` | str | Path to pretrained weights for inference or fine-tuning |

---

## Available Configurations

### RGB-only Baseline Methods (Illuminant Estimation)

| Config | Model | Description |
|--------|-------|-------------|
| `GW.py` | Gray World | Classic statistical method (`mink_norm=1`) |
| `WP.py` | White Patch | Max-RGB method |
| `SoG.py` | Shades of Gray | Generalized Gray World (`mink_norm=5`) |
| `GGW.py` | General Gray World | First-order Gray Edge |
| `GE1.py` | Gray Edge (1st order) | First-order edge-based method |
| `GE2.py` | Gray Edge (2nd order) | Second-order edge-based method |
| `ConvMean.py` | ConvMean | Learning-based illuminant estimation |
| `FC4.py` | FC4 | SqueezeNet-based illuminant estimation |
| `QuasiUnsupervised.py` | Quasi-Unsupervised | Self-supervised AWB method |
| `QuasiUnsupervised_ft.py` | QU Fine-tuned | Quasi-Unsupervised with fine-tuning |

### Multispectral Methods (RGB + MS)

| Config | Model | Description |
|--------|-------|-------------|
| `SpectralConvMean.py` | Spectral ConvMean | ConvMean extended with multispectral input |
| `SpectralFC4.py` | Spectral FC4 | FC4 extended with multispectral input |
| `SpectralLPIENet.py` | Spectral LPIENet | Joint image-to-image model with RGB+MS fusion |
| `SpectralLPIENet_small.py` | Spectral LPIENet (small) | Lightweight version of SpectralLPIENet |
| `SpectralCmKAN_light.py` | Spectral CmKAN (light) | Lightweight KAN-based spectral model |

---

## Pre-trained Models

Pre-trained weights are available in the `pretrained/` folder, organized by camera:

```
pretrained/
├── CanonR5/
├── GooglePixel3/
├── HuaweiMate20Pro/
├── iPhoneXsMax/
├── NikonZf/
├── SamsungGalaxyNote9/
└── SonyAlpha9III/
```

Each camera folder contains weights for both aligned and misaligned data scenarios:

| Weight File | Description |
|-------------|-------------|
| `FC4.pth` | FC4 baseline (RGB-only) |
| `ConvMean.pth` | ConvMean baseline (RGB-only) |
| `QU_ft.pth` | Quasi-Unsupervised fine-tuned |
| `SpectralFC4.pth` | Spectral FC4 (aligned data) |
| `SpectralFC4_misaligned.pth` | Spectral FC4 (fine-tuned on misaligned data) |
| `SpectralConvMean.pth` | Spectral ConvMean (aligned data) |
| `SpectralConvMean_misaligned.pth` | Spectral ConvMean (misaligned data) |
| `SpectralLPIENet.pth` | Spectral LPIENet (aligned data) |
| `SpectralLPIENet_misaligned.pth` | Spectral LPIENet (misaligned data) |
| `SpectralLPIENet_small.pth` | Spectral LPIENet small (aligned data) |
| `SpectralLPIENet_small_misaligned.pth` | Spectral LPIENet small (misaligned data) |
| `SpectralCmKAN_light.pth` | Spectral CmKAN light (aligned data) |
| `SpectralCmKAN_light_misaligned.pth` | Spectral CmKAN light (misaligned data) |

---

## Example: Testing with Pretrained Weights

To test a pretrained model, set `train = False` and specify the path to the pretrained weights:

```python
cfg = dict(
    exp_name = "CanonR5_SpectralFC4_test",
    train = False,
    test = True,
    
    # ... other settings ...
    
    pretrained_weights = "pretrained/CanonR5/SpectralFC4.pth",  # aligned data
    # pretrained_weights = "pretrained/CanonR5/SpectralFC4_misaligned.pth",  # misaligned data
)
```

## Example: Fine-tuning on Misaligned Data

To fine-tune a model trained on aligned data for misaligned scenarios:

```python
cfg = dict(
    exp_name = "CanonR5_SpectralFC4_finetune_misaligned",
    train = True,
    test = True,
    
    # Use misaligned data splits
    train_list = "data/data_splits/misaligned/train.txt",
    val_list = "data/data_splits/misaligned/val.txt",
    test_list = "data/data_splits/misaligned/test.txt",
    
    # Start from aligned pretrained weights
    pretrained_weights = "pretrained/CanonR5/SpectralFC4.pth",
)
```

---

## Customizing Configurations

To create a custom configuration:

1. Copy an existing config file that closely matches your setup
2. Modify the `exp_name` to a unique identifier
3. Adjust model, dataset, and training parameters as needed
4. Ensure `dataset_root` points to the correct location
5. Run with `python run.py conf/your_config.py`
