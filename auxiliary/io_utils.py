import h5py
import numpy as np
from scipy import io
import torch
import torch.nn.functional as F
import cv2
import json
import os
from functools import lru_cache

import ipdb

# Default path for Zurich homographies
DEFAULT_HOMOGRAPHIES_PATH = "data/Zurich_homographies(512x512)"
HOMOGRAPHY_REF_SIZE = 512  # Reference size for which homographies were computed


def read_homographies(path=DEFAULT_HOMOGRAPHIES_PATH):
    """Load homography matrices from a directory of .npy files."""
    file_names = sorted(os.listdir(path))
    homographies = np.array([np.load(os.path.join(path, f)) for f in file_names if f.endswith('.npy')])
    return homographies


def scale_homography(M, src_size, dst_size=HOMOGRAPHY_REF_SIZE):
    """
    Scale a homography matrix designed for one image size to work with another.
    
    The homography M is designed for images of size dst_size x dst_size.
    We need to scale it to work with images of size src_size.
    
    Args:
        M: 3x3 homography matrix
        src_size: tuple (h, w) of the source image size
        dst_size: int, the reference size for which M was computed (default 512)
    
    Returns:
        M_scaled: 3x3 scaled homography matrix
    """
    src_h, src_w = src_size
    
    # Scale factors
    scale_x = src_w / dst_size
    scale_y = src_h / dst_size
    
    # Scaling matrices
    S = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    S_inv = np.array([
        [1/scale_x, 0, 0],
        [0, 1/scale_y, 0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Scale the homography: M_scaled = S @ M @ S_inv
    M_scaled = S @ M @ S_inv
    
    return M_scaled


def warp_and_crop(img, M):
    """
    Warps and crops an image using the given perspective matrix.
    
    Supports:
        - Grayscale (H, W)
        - RGB or multichannel (H, W, C)
        - Channel-first multichannel (C, H, W)

    Parameters:
        img (np.ndarray): Input image
        M (np.ndarray): 3x3 perspective transformation matrix

    Returns:
        np.ndarray: Warped and cropped image in the same shape format as input
    """
    # Handle channel-first (C, H, W)
    if img.ndim == 3 and img.shape[0] < img.shape[1] and img.shape[0] < img.shape[2]:
        img = np.moveaxis(img, 0, -1)  # (C, H, W) → (H, W, C)
        channel_first = True
    else:
        channel_first = False

    h, w = img.shape[:2]

    # Warp the image (single or multi-channel)
    if img.ndim == 2:  # Grayscale
        warped = cv2.warpPerspective(img, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
    else:
        channels = []
        for i in range(img.shape[2]):
            warped_channel = cv2.warpPerspective(img[..., i], M, (w, h),
                                                 flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=0)
            channels.append(warped_channel)
        warped = np.stack(channels, axis=-1)

    # Get warped corner positions
    corners = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)
    dst_pts = cv2.perspectiveTransform(corners, M.astype(np.float32)).reshape(-1, 2)

    # Compute bounding box
    x_min = max(0, int(np.ceil(np.max([dst_pts[0,0], dst_pts[3,0]]))))
    x_max = min(w, int(np.floor(np.min([dst_pts[1,0], dst_pts[2,0]]))))
    y_min = max(0, int(np.ceil(np.max([dst_pts[0,1], dst_pts[1,1]]))))
    y_max = min(h, int(np.floor(np.min([dst_pts[2,1], dst_pts[3,1]]))))
    
    # Crop the warped image
    cropped = warped[y_min:y_max, x_min:x_max]
    
    # Ensure the cropped image has the same size as the original one
    cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    # Restore original channel order
    if channel_first and cropped.ndim == 3:
        cropped = np.moveaxis(cropped, -1, 0)  # back to (C, H, W)

    return cropped


def read_h5(path, key="spec"):
        with h5py.File(path, "r") as f:
            spec = np.array(f[key]).astype(np.float32)
            # If "wvs" exists, load it
            if "wvs" in f.keys():
                wavelengths = np.array(f["wvs"]).astype(np.float32)
            else:
                wavelengths = np.arange(400, 701, 10).astype(np.float32)  # Default wavelengths from 400nm to 700nm with 10nm interval

        return spec, wavelengths


def save_h5(path, spec, wvs):
    # Create the visualization directory
    _, h,w = spec.shape 

    # Save to h5 file
    hf = h5py.File(path, "w")
    hf.create_dataset("spec", data=spec.detach().cpu().numpy())
    hf.create_dataset("wvs", data=wvs)

    hf.close()

def load_rgb_image(path, bit_depth=12):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / (2**bit_depth - 1)
    img =img.transpose(2,0,1) # HWC to CHW

    return img


@lru_cache(maxsize=256)
def load_metadata(path):
    with open(path, "r") as f:
        metadata = json.load(f)

    for key in metadata:
        if isinstance(metadata[key], list):
            metadata[key] = np.array(metadata[key]).astype(np.float32)

    return metadata


def read_ssf(path):
    """Read spectral sensitivity function from h5 file."""
    with h5py.File(path, 'r') as f:
        camera_name = str(f['camera_name'][()]).replace("b'", "").replace("'", "")
        wvs = np.array(f['wvs'])
        
        if "rgb" in f.keys():
            filters = np.array(f['rgb'])
        elif "spec" in f.keys():
            filters = np.array(f['spec'])
    return camera_name, wvs, filters


def read_reflectance_h5(path):
    """Read reflectance data from h5 file (for KAUST/BJTU spectral datasets)."""
    with h5py.File(path, 'r') as f:
        refl = np.array(f['spec'])
        wvs = np.array(f['wvs'])
        rois = np.array(f['rois']) if "rois" in f.keys() else None
    
    refl = (refl - refl.min()) / (refl.max() - refl.min())
    
    if rois is not None:
        for roi in rois:
            x, y, d_x, d_y = roi
            refl[:, y:y+d_y, x:x+d_x] = 0
    return refl, wvs


def get_ssf_path_for_camera(camera_name, ssf_base_path="data/ImageEngineering_SSFs/h5"):
    """Find the SSF file path for a given camera name."""
    camera_name_normalized = camera_name.upper().replace("_", " ")
    ssf_files = os.listdir(ssf_base_path)
    
    for ssf_file in ssf_files:
        ssf_name_normalized = ssf_file.replace(".h5", "").upper().replace("_", " ")
        if camera_name_normalized == ssf_name_normalized or \
           camera_name_normalized.replace(" ", "") == ssf_name_normalized.replace(" ", ""):
            return os.path.join(ssf_base_path, ssf_file)
    
    # Try partial matching
    for ssf_file in ssf_files:
        ssf_name_normalized = ssf_file.replace(".h5", "").upper().replace("_", " ")
        if camera_name_normalized in ssf_name_normalized or ssf_name_normalized in camera_name_normalized:
            return os.path.join(ssf_base_path, ssf_file)
    
    return None