import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import h5py
import numpy as np
import json
from colour import SpragueInterpolator, SpectralShape, SDS_ILLUMINANTS, MSDS_CMFS, XYZ_to_sRGB, XYZ_to_xy, xy_to_CCT, SDS_COLOURCHECKERS
from colour.colorimetry import blackbody_spectral_radiance
from scipy.ndimage import zoom
from auxiliary.color_utils import CCT_Robertson 



D65 = SDS_ILLUMINANTS['D65'].align(SpectralShape(400, 700, 10)).values
E = SDS_ILLUMINANTS['E'].align(SpectralShape(400, 700, 10)).values
XYZ_CMF = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].align(SpectralShape(400, 700, 10)).values


def read_ssf(path):
    with h5py.File(path, 'r') as f:
        camera_name = str(f['camera_name'][()]).replace("b'", "").replace("'", "")
        wvs = np.array(f['wvs'])
        
        if "rgb" in f.keys():
            filters = np.array(f['rgb'])
        elif "spec" in f.keys():
            filters = np.array(f['spec'])
    return camera_name, wvs, filters

def interpolate(spec, old_wvs, new_wvs):
    sprague_interpolator = SpragueInterpolator(old_wvs, spec)
    spec = sprague_interpolator(new_wvs)
    
    return spec

def render_image(refl, ill, ssf_rgb):
    c, h, w = refl.shape

    rad = refl * ill[:, None, None]
    
    rad = rad.reshape(c, h*w)
    img = ssf_rgb.T @ rad
    img = img.reshape(ssf_rgb.shape[1], h, w).transpose(1, 2, 0)
    
    ill_rgb = ssf_rgb.T @ ill

    
    img = img / ill_rgb.max()
    img = np.clip(img, 0, 1)

    return img, ill_rgb/np.linalg.norm(ill_rgb)


def read_h5(path):
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

def save_h5(path,spec, wvs):
    with h5py.File(path, 'w') as f:
        f.create_dataset('spec', data=spec)
        f.create_dataset('wvs', data=wvs)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def xyz2srgb(img):
    h,w,c = img.shape
    
    img = img.reshape(h*w, c)
    img = XYZ_to_sRGB(img)
    img = img.reshape(h,w,c)

    img = img.clip(0,1)

    return img

def compute_correction_matrix(ssf_rgb, temp=None, ill=None):
    assert (temp is not None) ^ (ill is not None), "Only one between (temp) and (ill) should be specified"
    if ill is not None:
        bb_spectrum = SDS_ILLUMINANTS[ill].align(SpectralShape(400, 700, 10)).values
    else:
        # Use colour library to compute the black body spectrum at the given temperature
        bb_spectrum = np.array([blackbody_spectral_radiance(wv, temp) for wv in np.arange(400, 701, 10)*1e-9])
    
    
    # Macbeth Color Checker reflectance spectra
    macbeth_spectra = np.array([SDS_COLOURCHECKERS['cc_ohta'][k].align(SpectralShape(400, 700, 10)).values for k in SDS_COLOURCHECKERS['cc_ohta'].keys()])
    # Compute the RGB values for the Macbeth Color Checker under the black body illuminant
    macbeth_rgb = (ssf_rgb.T @ (macbeth_spectra.T * bb_spectrum[:, None]))
    illum = (ssf_rgb.T @ bb_spectrum)
    macbeth_rgb = macbeth_rgb / illum.max()
    macbeth_rgb = (macbeth_rgb * (2**12 -1)).round() / (2**12 -1)
    illum = ((illum/illum.max()) * (2**12 -1)).round() / (2**12 -1)
    macbeth_rgb = macbeth_rgb / (illum/np.linalg.norm(illum))[:,None]

    # Compute the XYZ values for the Macbeth Color Checker under D65
    macbeth_xyz = (XYZ_CMF.T @ (macbeth_spectra.T * D65[:, None]))
    d65_max = (XYZ_CMF.T @ D65).max()
    macbeth_xyz = macbeth_xyz / d65_max
    macbeth_xyz = (macbeth_xyz * (2**12 -1)).round() / (2**12 -1)
    # Compute the correction matrix from RGB to XYZ:  = RGB(nx3) @ CORR(3x3) = XYZ(nx3)
    correction_matrix, _, _, _ = np.linalg.lstsq(macbeth_rgb.T, macbeth_xyz.T, rcond=None)
    integrated_cct = np.diag((illum/np.linalg.norm(illum))**(-1)) @ correction_matrix
    return correction_matrix, integrated_cct

def compute_illuminant_correction_matrix(ssf_rgb):
    ill_list = []
    xyz_ill_list = []
    for temp in np.arange(2500, 6501, 500):
        bb_spectrum = np.array([blackbody_spectral_radiance(wv, temp) for wv in np.arange(400, 701, 10)*1e-9])
        illum = (ssf_rgb.T @ bb_spectrum)
        illum = illum / np.linalg.norm(illum)

        illum_xyz = (XYZ_CMF.T @ bb_spectrum)
        illum_xyz = illum_xyz / np.linalg.norm(illum_xyz)

        ill_list.append(illum)
        xyz_ill_list.append(illum_xyz)

    ill_list = np.stack(ill_list, axis=0)
    xyz_ill_list = np.stack(xyz_ill_list, axis=0)

    # Compute the correction matrix from RGB to XYZ:  = ILL(nx3) @ CORR(3x3) = XYZ(nx3)
    correction_matrix, _, _, _ = np.linalg.lstsq(ill_list, xyz_ill_list, rcond=None)

    return correction_matrix

def interpolate_ccm(ill_rgb, cct1, cct2, ill_cct, temp1=2500, temp2=6500):
    # Correlated color temperature of the illuminant
    xyz = ill_rgb @ ill_cct

    status, cct = CCT_Robertson().XYZtoCorColorTemp(xyz)
    if status != 0:
        print("Warning: CCT calculation failed!")
        
    g = (cct**(-1) - temp2**(-1)) / (temp1**(-1) - temp2**(-1))

    ccm = g * cct1 + (1-g) * cct2

    return ccm

def resize(x, out_h=None, out_w=None, scale=None, order=3):
    assert (out_h is not None and out_w is not None) or (scale is not None), "Either (out_h, out_w) or scale should be specified"
    assert not (out_h is not None and out_w is not None and scale is not None), "Either (out_h, out_w) or scale should be specified, not both"

    if scale is not None:
        out_h = int(x.shape[1] * scale)
        out_w = int(x.shape[2] * scale)    

    zoom_h = out_h / x.shape[1]
    zoom_w = out_w / x.shape[2]
    return zoom(x, (1, zoom_h, zoom_w), order=order)  # order=3 = bicubic

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
    original_shape = img.shape
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
    dst_pts = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

    # Compute bounding box
    x_min = max(0, int(np.ceil(np.max([dst_pts[0,0], dst_pts[3,0]]))))
    x_max = min(w, int(np.floor(np.min([dst_pts[1,0], dst_pts[2,0]]))))
    y_min = max(0, int(np.ceil(np.max([dst_pts[0,1], dst_pts[1,1]]))))
    y_max = min(h, int(np.floor(np.min([dst_pts[2,1], dst_pts[3,1]]))))
    
    # Crop the warped image
    cropped = warped[y_min:y_max, x_min:x_max]

    # Restore original channel order
    if channel_first and cropped.ndim == 3:
        cropped = np.moveaxis(cropped, -1, 0)  # back to (C, H, W)

    return cropped

def read_homographies(path):
    file_names = os.listdir(path)
    homographies = np.array([np.load(os.path.join(path, f)) for f in file_names])

    return homographies

if __name__ == "__main__":
    random_seed = 42
    misaligned = False
    HQ = False
    cameras = [
        "Google Pixel 3", 
        "iPhone XS Max",
        "Samsung Galaxy Note 9", 
        "Huawei Mate 20 Pro", 
        "Canon R5", 
        "Nikon Zf", 
        "Sony Alpha 9 III",
        "Samsung Galaxy Tab 9",
    ]

    np.random.seed(random_seed)
    homographies = read_homographies(os.path.join("data", "Zurich_homographies(512x512)"))


    for spectral_dataset_name, spectral_dataset_path in zip(["KAUST_AWB", "BJTU-UVA"], 
                                                          [os.path.join("data", "KAUST_AWB", "h5"), 
                                                           os.path.join("data", "BJTU-UVA", "31bands_h5")]):
        
        print(f"Processing spectral dataset: {spectral_dataset_name}...")

        dataset_code = "K" if spectral_dataset_name == "KAUST_AWB" else "B"

        rgb_ssfs_dataset_path = os.path.join("data", "ImageEngineering_SSFs", "h5")
        ms_ssfs_dataset_path = os.path.join("data", "Multispectral_SSFs", "h5")
        illums_dataset_path = os.path.join("data", "SFU_measured_with_sources_illums", "measured_with_sources.illum.npy")
        destination_path = os.path.join("data", "MobileSpectralCCDataset")
        os.makedirs(os.path.join(destination_path), exist_ok=True)
        
        scenes = sorted(os.listdir(spectral_dataset_path)) 
        ms_ssfs = sorted(os.listdir(ms_ssfs_dataset_path))
        rgb_ssfs = sorted(os.listdir(rgb_ssfs_dataset_path))
        target_wvs = np.arange(400, 701, 10)

        illums, illums_wvs = np.load(illums_dataset_path), np.arange(380, 781, 4)

        os.makedirs(os.path.join(destination_path,"GT","xyz_scenes"), exist_ok=True)
        os.makedirs(os.path.join(destination_path,"GT","srgb_scenes"), exist_ok=True)
        os.makedirs(os.path.join(destination_path,"GT","illums_spd"), exist_ok=True)    
        
        # Save GT images
        print("Generating GT images...")
        for n_scene, scene in enumerate(scenes):
            refl, _ = read_h5(os.path.join(spectral_dataset_path, scene))
            
            gt_img, _ = render_image(refl, D65, XYZ_CMF)
            gt_img_srgb = xyz2srgb(gt_img)
            cv2.imwrite(os.path.join(destination_path, "GT", "xyz_scenes", f"SC{n_scene:03d}_{dataset_code}.png"), cv2.cvtColor(((gt_img * (2**12 -1)).astype(np.uint16)), cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(destination_path, "GT", "srgb_scenes", f"SC{n_scene:03d}_{dataset_code}.png"), cv2.cvtColor(((gt_img_srgb * 255).astype(np.uint8)), cv2.COLOR_RGB2BGR))

        # Save illuminants SPDs GT
        print("Generating illuminants SPDs...")
        for n_ill, ill in enumerate(illums):
            ill = interpolate(ill, illums_wvs, target_wvs)
            save_h5(os.path.join(destination_path, "GT", "illums_spd", f"ILL{n_ill:03d}.h5"), ill, target_wvs)

        # Create images for each rgb camera
        print("Generating images for each camera...")
        for ssf in rgb_ssfs:
            if all([c.upper().replace(" ", "_") not in ssf.upper().replace(" ", "_") for c in cameras]):
                continue
            print(f"Processing {ssf}...")
            for n_scene, scene in enumerate(scenes):
                refl, _ = read_h5(os.path.join(spectral_dataset_path, scene))

                camera_name, ssf_wvs, ssf_rgb = read_ssf(os.path.join(rgb_ssfs_dataset_path, ssf))
                ssf_rgb = np.stack([interpolate(ssf_rgb[i, :], ssf_wvs, target_wvs) for i in range(3)], axis=1)
                os.makedirs(os.path.join(destination_path, camera_name.upper().replace(" ", "_"), "imgs"), exist_ok=True)
                os.makedirs(os.path.join(destination_path, camera_name.upper().replace(" ", "_"), "metadata"), exist_ok=True)
                os.makedirs(os.path.join(destination_path, camera_name.upper().replace(" ", "_"), "gt_raw"), exist_ok=True)

                raw_gt, _ = render_image(refl, E, ssf_rgb)
                ill_cct = compute_illuminant_correction_matrix(ssf_rgb)
                _ ,cct_base = compute_correction_matrix(ssf_rgb, ill="E")
                cct1, _ = compute_correction_matrix(ssf_rgb, temp=2500)
                cct2, _ = compute_correction_matrix(ssf_rgb, temp=6500)
                cv2.imwrite(os.path.join(destination_path, camera_name.upper().replace(" ", "_"), "gt_raw", f"SC{n_scene:03d}_{dataset_code}.png"), cv2.cvtColor(((raw_gt * (2**12 -1)).astype(np.uint16)), cv2.COLOR_RGB2BGR))
                    
                for n_ill, ill in enumerate(illums):
                    # if n_scene == 0:
                    ill = interpolate(ill, illums_wvs, target_wvs)
                
                    raw_img, ill_rgb = render_image(refl, ill, ssf_rgb)
                    cv2.imwrite(os.path.join(destination_path, camera_name.upper().replace(" ", "_"), "imgs", f"SC{n_scene:03d}_{dataset_code}_ILL{n_ill:03d}.png"), cv2.cvtColor(((raw_img * (2**12 -1)).astype(np.uint16)), cv2.COLOR_RGB2BGR))
                    cct_interp = interpolate_ccm(ill_rgb, cct1, cct2, ill_cct)

                    if n_scene == 0:
                        metadata = {
                            "illuminant_rgb": ill_rgb.tolist(),
                            "ill_cct": ill_cct.tolist(),
                            "cct_base": cct_base.tolist(),
                            "cct_2500K": cct1.tolist(),
                            "cct_6500K": cct2.tolist(),
                            "cct_interp": cct_interp.tolist(),
                        }
                        save_json(os.path.join(destination_path, camera_name.upper().replace(" ", "_"), "metadata", f"ILL{n_ill:03d}.json"), metadata)
                    

        # Create images for each spectral camera
        print("Generating images for each multispectral camera...")
        for ssf in ms_ssfs:
            print(f"Processing {ssf}...")
            camera_name, ssf_wvs, ssf_ms = read_ssf(os.path.join(ms_ssfs_dataset_path, ssf))

            if camera_name.upper() == "HQ SPECTRAL":
                continue

            if HQ:
                camera_name += "_HIGHRES"

            if misaligned:
                camera_name += "_MISALIGNED"

            
            ssf_ms = np.stack([interpolate(ssf_ms[i, :], ssf_wvs, target_wvs) for i in range(ssf_ms.shape[0])], axis=1)
            os.makedirs(os.path.join(destination_path, camera_name.upper().replace(" ", "_"), "imgs"), exist_ok=True)
            os.makedirs(os.path.join(destination_path, camera_name.upper().replace(" ", "_"), "illums_spd"), exist_ok=True)
            os.makedirs(os.path.join(destination_path, camera_name.upper().replace(" ", "_"), "gt_raw"), exist_ok=True)
            for n_scene, scene in enumerate(scenes):                                           
                refl, refl_wvs = read_h5(os.path.join(spectral_dataset_path, scene))
                if misaligned:
                    # M = np.array([[1.09945169e+00, 1.28540198e-03, -2.13997299e+01],
                    #             [-5.04230449e-03, 1.11190594e+00, -3.77604344e+01],
                    #             [-3.03017375e-05, -1.85108750e-06, 1.00000000e+00]]) # Computed from Zurich dataset (resized to 512x512)
                    # Sample a random homography from the Zurich dataset
                    M = homographies[np.random.randint(0, len(homographies))]
                    
                    refl = warp_and_crop(refl.transpose(1,2,0), M).transpose(2,0,1)
                    if not HQ:
                        refl = resize(refl, out_h=64, out_w=64, order=3)  
                else:
                    if not HQ:
                        refl = resize(refl, scale=0.125, order=3)

                for n_ill, ill in enumerate(illums):

                    ill = interpolate(ill, illums_wvs, target_wvs)
                    
                    raw_img, ill_spd = render_image(refl, ill, ssf_ms)

                    # # Quantize to 10 bits
                    raw_img = (raw_img * (2**10 -1)).astype(np.uint16) / (2**10 -1)
                    ill_spd = (ill_spd * (2**10 -1)).astype(np.uint16) / (2**10 -1)

                    save_h5(os.path.join(destination_path, camera_name.upper().replace(" ", "_"), "imgs", f"SC{n_scene:03d}_{dataset_code}_ILL{n_ill:03d}.h5"), raw_img.transpose(2,0,1), np.stack([target_wvs[np.argmax(ssf_ms[:,i])] for i in range(len(ssf_ms[0]))], axis=0))

                    if n_scene == 0:
                        save_h5(os.path.join(destination_path, camera_name.upper().replace(" ", "_"), "illums_spd", f"ILL{n_ill:03d}.h5"), ill_spd, np.stack([target_wvs[np.argmax(ssf_ms[:,i])] for i in range(len(ssf_ms[0]))], axis=0))

                raw_gt, _ = render_image(refl, E, ssf_ms)
                save_h5(os.path.join(destination_path, camera_name.upper().replace(" ", "_"), "gt_raw", f"SC{n_scene:03d}_{dataset_code}.h5"), raw_gt.transpose(2,0,1), np.stack([target_wvs[np.argmax(ssf_ms[:,i])] for i in range(len(ssf_ms[0]))], axis=0))
