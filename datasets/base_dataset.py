import os
import sys
import numpy as np
import h5py
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from torch.utils.data import Dataset
from auxiliary.io_utils import load_rgb_image, read_h5, load_metadata, get_ssf_path_for_camera
from auxiliary.color_utils import SpectralImageRenderer
from auxiliary.color_utils import XYZ_CMF_SPECTRUM, E_SPECTRUM, D65_SPECTRUM

import time

import ipdb


# Default paths for on-the-fly generation
DEFAULT_SSF_RGB_PATH = "data/ImageEngineering_SSFs/h5"
DEFAULT_SSF_MS_PATH = "data/Multispectral_SSFs/h5"
DEFAULT_ILLUMINANTS_PATH = "data/SFU_measured_with_sources_illums/measured_with_sources.illum.npy"
DEFAULT_HOMOGRAPHIES_PATH = "data/Zurich_homographies(512x512)"
DEFAULT_SPECTRAL_DATASETS = {
    "K": "data/KAUST_AWB/h5",
    "B": "data/BJTU-UVA/31bands_h5"
}


class BaseDataset(Dataset):
    def __init__(self, dataset_root, files_list, is_train=True, seed=42):
        super(BaseDataset, self).__init__()

        self.dataset_root = dataset_root
        self.files_list = self.load_scenes(files_list)
        self.is_train = is_train
        self.seed = seed
        
        # For on-the-fly generation
        self.renderer = None
        self.generate_on_the_fly = False
        
        # Parse unique scenes to pre-assign illuminants for val/test
        self._scene_illuminant_map = {}
        if not is_train:
            self._preassign_illuminants()

        # self.files_list = self.files_list[:int(0.1*len(self.files_list))]  # For quick testing, remove this line for full dataset
        # self.files_list = self.files_list[:100]  # For quicker testing, remove this line for full dataset
        
    def load_scenes(self, scenes_file):
        with open(scenes_file, "r") as f:
            scenes = [line.strip() for line in f.readlines()]
        return scenes
    
    def _preassign_illuminants(self):
        """Pre-assign illuminants for validation/test to ensure consistency across epochs."""
        # This will be called after renderer is initialized in subclasses
        pass
    
    def _get_illuminant_idx(self, idx, scene_key):
        """Get illuminant index for a sample. Random for train, deterministic for val/test."""
        if self.renderer is None or self.renderer.get_num_illuminants() == 0:
            return 0
            
        n_illuminants = self.renderer.get_num_illuminants()
        
        if self.is_train:
            # Random illuminant for training
            return np.random.randint(0, n_illuminants)
        else:
            # Deterministic illuminant for val/test based on scene and seed
            if scene_key not in self._scene_illuminant_map:
                # Use hash of scene_key and seed for deterministic selection
                rng = np.random.RandomState(hash(scene_key) % (2**32) + self.seed)
                self._scene_illuminant_map[scene_key] = rng.randint(0, n_illuminants)
            return self._scene_illuminant_map[scene_key]
    
    def _get_homography_idx(self, idx, scene_key, n_homographies):
        """Get homography index for a sample. Random for train, deterministic for val/test."""
        if n_homographies == 0:
            return 0
            
        if self.is_train:
            # Random homography for training
            return np.random.randint(0, n_homographies)
        else:
            # Deterministic homography for val/test based on scene and seed
            # Use a separate map to avoid collision with illuminant assignment
            map_key = f"homography_{scene_key}"
            if map_key not in self._scene_illuminant_map:
                # Use hash of scene_key and seed for deterministic selection
                rng = np.random.RandomState(hash(map_key) % (2**32) + self.seed)
                self._scene_illuminant_map[map_key] = rng.randint(0, n_homographies)
            return self._scene_illuminant_map[map_key]
    
    def __len__(self):
        return len(self.files_list)
    
    def __getitem__(self, idx):
        raise NotImplementedError("This method should be overridden by subclasses")

class RGBDataset(BaseDataset):
    def __init__(self, dataset_root, files_list, rgb_camera, gt_type, 
                 is_train=True, seed=42,
                 ssf_path=None, illuminants_path=None, spectral_datasets=None):
        super(RGBDataset, self).__init__(dataset_root, files_list, is_train=is_train, seed=seed)
        self.rgb_camera = rgb_camera
        self.gt_type = gt_type
        
        # Check if pre-generated images exist
        sample_file = self.files_list[0] if len(self.files_list) > 0 else None
        if sample_file:
            img_path = os.path.join(self.dataset_root, self.rgb_camera, "imgs", sample_file + ".png")
            self.generate_on_the_fly = not os.path.exists(img_path)
        
        # Initialize renderer for on-the-fly generation
        if self.generate_on_the_fly:
            ssf_path = ssf_path or get_ssf_path_for_camera(rgb_camera, DEFAULT_SSF_RGB_PATH)
            illuminants_path = illuminants_path or DEFAULT_ILLUMINANTS_PATH
            spectral_datasets = spectral_datasets or DEFAULT_SPECTRAL_DATASETS
            
            if ssf_path is None:
                raise ValueError(f"Could not find SSF for camera {rgb_camera}")
            
            self.renderer = SpectralImageRenderer(
                ssf_path=ssf_path,
                spectral_datasets=spectral_datasets,
                illuminants_path=illuminants_path
            )
            print(f"[RGBDataset] On-the-fly generation enabled for camera {rgb_camera}")

    def __getitem__(self, idx):
        file_name = self.files_list[idx]
        
        # Parse scene name and dataset code
        # Format: SC001_K_ILL000 or SC001_B_ILL000
        parts = file_name.split("_")
        scene_name = parts[0]  # SC001
        dataset_code = parts[1]  # K or B
        
        if self.generate_on_the_fly:
            # Generate on-the-fly
            scene_key = f"{scene_name}_{dataset_code}"
            illuminant_idx = self._get_illuminant_idx(idx, scene_key)
            
            rgb_image, metadata = self.renderer.render_rgb(scene_name, dataset_code, illuminant_idx)
            
            # Generate GT
            if self.gt_type == "raw":
                gt_image = self.renderer.render_gt_rgb(scene_name, dataset_code)
            else:
                # For xyz/srgb GT, we still need to load from pre-generated files
                gt_path = os.path.join(self.dataset_root, "GT", self.gt_type + "_scenes", f"{scene_name}_{dataset_code}.png")
                if os.path.exists(gt_path):
                    gt_image = load_rgb_image(path=gt_path, bit_depth=12)
                else:
                    # Fall back to raw GT
                    gt_image = self.renderer.render_gt_rgb(scene_name, dataset_code)
            
            # Update file_name to include illuminant info
            file_name = f"{scene_name}_{dataset_code}_ILL{illuminant_idx:03d}"
        else:
            # Load from pre-generated files
            scene_name_full = file_name.split("_ILL")[0]  # SC001_K
            ill_name = "ILL" + file_name.split("_ILL")[1]

            # Load RGB image 
            rgb_image = load_rgb_image(path=os.path.join(self.dataset_root, self.rgb_camera, "imgs", file_name+".png"), bit_depth=12)

            # Load metadata
            metadata = load_metadata(path=os.path.join(self.dataset_root, self.rgb_camera, "metadata", ill_name+".json"))

            # Load GT
            if self.gt_type == "raw":
                gt_image = load_rgb_image(os.path.join(self.dataset_root, self.rgb_camera, "gt_raw", scene_name_full+".png"), bit_depth=12)
            else:
                gt_image = load_rgb_image(path=os.path.join(self.dataset_root, "GT", self.gt_type+"_scenes", scene_name_full+".png"), bit_depth=12)

        sample = {
            "file_name": file_name,
            "rgb_image": rgb_image,
            "gt_image": gt_image,
            "metadata": metadata
        }

        return sample

class MSDataset(BaseDataset):
    def __init__(self, dataset_root, files_list, spectral_camera, gt_type,
                 is_train=True, seed=42,
                 ssf_path=None, illuminants_path=None, spectral_datasets=None,
                 misaligned=False, homographies_path=None):
        super(MSDataset, self).__init__(dataset_root, files_list, is_train=is_train, seed=seed)
        self.spectral_camera = spectral_camera
        self.gt_type = gt_type
        self.misaligned = misaligned

        self.means = []
        self.stds = []
        
        # Check if pre-generated images exist
        sample_file = self.files_list[0] if len(self.files_list) > 0 else None
        if sample_file:
            img_path = os.path.join(self.dataset_root, self.spectral_camera, "imgs", sample_file + ".h5")
            self.generate_on_the_fly = not os.path.exists(img_path)
        
        # Initialize renderer for on-the-fly generation
        if self.generate_on_the_fly:
            ssf_path = ssf_path or get_ssf_path_for_camera(spectral_camera, DEFAULT_SSF_MS_PATH)
            illuminants_path = illuminants_path or DEFAULT_ILLUMINANTS_PATH
            spectral_datasets = spectral_datasets or DEFAULT_SPECTRAL_DATASETS
            homographies_path = homographies_path or DEFAULT_HOMOGRAPHIES_PATH
            
            if ssf_path is None:
                raise ValueError(f"Could not find SSF for camera {spectral_camera}")
            
            self.renderer = SpectralImageRenderer(
                ssf_path=ssf_path,
                spectral_datasets=spectral_datasets,
                illuminants_path=illuminants_path,
                misaligned=misaligned,
                homographies_path=homographies_path
            )
            print(f"[MSDataset] On-the-fly generation enabled for camera {spectral_camera}" + 
                  (" (misaligned)" if misaligned else ""))

    def __getitem__(self, idx):
        file_name = self.files_list[idx]
        
        # Parse scene name and dataset code
        parts = file_name.split("_")
        scene_name = parts[0]  # SC001
        dataset_code = parts[1]  # K or B
        
        if self.generate_on_the_fly:
            # Generate on-the-fly
            scene_key = f"{scene_name}_{dataset_code}"
            illuminant_idx = self._get_illuminant_idx(idx, scene_key)
            
            # Get homography index for misalignment if enabled
            homography_idx = None
            if self.misaligned and self.renderer is not None:
                n_homographies = self.renderer.get_num_homographies()
                if n_homographies > 0:
                    homography_idx = self._get_homography_idx(idx, scene_key, n_homographies)
            
            ms_image, ms_wvs, ms_illum_spd = self.renderer.render_ms(scene_name, dataset_code, illuminant_idx, homography_idx)
            ill_wvs = ms_wvs.copy()
            
            # Generate GT (GT is always aligned, so no homography applied)
            gt_ms, _ = self.renderer.render_gt_ms(scene_name, dataset_code)
            
            # For xyz/srgb GT, load from pre-generated files
            gt_path = os.path.join(self.dataset_root, "GT", self.gt_type + "_scenes", f"{scene_name}_{dataset_code}.png")
            if os.path.exists(gt_path):
                gt_image = load_rgb_image(path=gt_path, bit_depth=12)
            else:
                gt_image = gt_ms[:3]  # Fallback to first 3 channels if no GT available
            
            file_name = f"{scene_name}_{dataset_code}_ILL{illuminant_idx:03d}"
        else:
            # Load from pre-generated files
            scene_name_full = file_name.split("_ILL")[0]
            ill_name = "ILL" + file_name.split("_ILL")[1]

            # Load MS image 
            ms_image, ms_wvs = read_h5(path=os.path.join(self.dataset_root, self.spectral_camera, "imgs", file_name+".h5"), key="spec")

            # Load metadata
            ms_illum_spd, ill_wvs = read_h5(path=os.path.join(self.dataset_root, self.spectral_camera, "illums_spd", ill_name+".h5"), key="spec")

            assert np.all(ill_wvs == ms_wvs), "Wavelengths of illuminant and MS camera do not match"

            # Load MS GT
            gt_ms, _ = read_h5(path=os.path.join(self.dataset_root, self.spectral_camera, "gt_raw", scene_name_full+".h5"), key="spec")

            # Load IMG GT
            gt_image = load_rgb_image(path=os.path.join(self.dataset_root, "GT", self.gt_type+"_scenes", scene_name_full+".png"), bit_depth=12)

        self.means.append(np.mean(ms_image, axis=(0,1)))
        self.stds.append(np.std(ms_image, axis=(0,1)))

        if self.means.__len__() % 100 == 0:
            print(f"[Dataset] Processed {self.means.__len__()} samples. Current mean: {np.mean(np.array(self.means), axis=0)}. Current std: {np.mean(np.array(self.stds), axis=0)}")

        sample = {
            "file_name": file_name,
            "ms_image": ms_image,
            "ms_wvs": ms_wvs,
            "ms_illum_spd": ms_illum_spd,
            "ill_wvs": ill_wvs,
            "gt_ms": gt_ms,
            "gt_image": gt_image
        }
        return sample

class RGBMSDataset(BaseDataset):
    def __init__(self, dataset_root, files_list, rgb_camera, spectral_camera, gt_type,
                 is_train=True, seed=42,
                 rgb_ssf_path=None, ms_ssf_path=None, illuminants_path=None, spectral_datasets=None,
                 misaligned=False, homographies_path=None):
        super(RGBMSDataset, self).__init__(dataset_root, files_list, is_train=is_train, seed=seed)
        self.rgb_camera = rgb_camera
        self.spectral_camera = spectral_camera
        self.gt_type = gt_type
        self.misaligned = misaligned

        self.rgb_means = []
        self.rgb_stds = []
        self.ms_means = []
        self.ms_stds = []
        
        # Check if pre-generated images exist for both RGB and MS
        sample_file = self.files_list[0] if len(self.files_list) > 0 else None
        if sample_file:
            rgb_img_path = os.path.join(self.dataset_root, self.rgb_camera, "imgs", sample_file + ".png")
            ms_img_path = os.path.join(self.dataset_root, self.spectral_camera, "imgs", sample_file + ".h5")
            self.generate_on_the_fly = not (os.path.exists(rgb_img_path) and os.path.exists(ms_img_path))
        
        # Initialize renderers for on-the-fly generation
        if self.generate_on_the_fly:
            rgb_ssf_path = rgb_ssf_path or get_ssf_path_for_camera(rgb_camera, DEFAULT_SSF_RGB_PATH)
            ms_ssf_path = ms_ssf_path or get_ssf_path_for_camera(spectral_camera, DEFAULT_SSF_MS_PATH)
            illuminants_path = illuminants_path or DEFAULT_ILLUMINANTS_PATH
            spectral_datasets = spectral_datasets or DEFAULT_SPECTRAL_DATASETS
            homographies_path = homographies_path or DEFAULT_HOMOGRAPHIES_PATH
            
            if rgb_ssf_path is None:
                raise ValueError(f"Could not find SSF for RGB camera {rgb_camera}")
            if ms_ssf_path is None:
                raise ValueError(f"Could not find SSF for MS camera {spectral_camera}")
            
            # RGB renderer - never misaligned (RGB is reference)
            self.rgb_renderer = SpectralImageRenderer(
                ssf_path=rgb_ssf_path,
                spectral_datasets=spectral_datasets,
                illuminants_path=illuminants_path,
                misaligned=False
            )
            # MS renderer - can be misaligned
            self.ms_renderer = SpectralImageRenderer(
                ssf_path=ms_ssf_path,
                spectral_datasets=spectral_datasets,
                illuminants_path=illuminants_path,
                misaligned=misaligned,
                homographies_path=homographies_path
            )
            # Use RGB renderer for illuminant selection
            self.renderer = self.rgb_renderer
            print(f"[RGBMSDataset] On-the-fly generation enabled for RGB camera {rgb_camera} and MS camera {spectral_camera}" +
                  (" (MS misaligned)" if misaligned else ""))
    
    def __getitem__(self, idx):
        file_name = self.files_list[idx]
        
        # Parse scene name and dataset code
        parts = file_name.split("_")
        scene_name = parts[0]  # SC001
        dataset_code = parts[1]  # K or B
        
        if self.generate_on_the_fly:
            # Generate on-the-fly
            scene_key = f"{scene_name}_{dataset_code}"
            illuminant_idx = self._get_illuminant_idx(idx, scene_key)
            
            # Get homography index for MS misalignment if enabled
            homography_idx = None
            if self.misaligned and self.ms_renderer is not None:
                n_homographies = self.ms_renderer.get_num_homographies()
                if n_homographies > 0:
                    homography_idx = self._get_homography_idx(idx, scene_key, n_homographies)
            
            # Render RGB (always aligned)
            rgb_image, metadata = self.rgb_renderer.render_rgb(scene_name, dataset_code, illuminant_idx)
            
            # Render MS (can be misaligned)
            ms_image, ms_wvs, ms_illum_spd = self.ms_renderer.render_ms(scene_name, dataset_code, illuminant_idx, homography_idx)
            ill_wvs = ms_wvs.copy()
            
            # Generate GT (GT is always aligned, so no homography applied)
            gt_ms, _ = self.ms_renderer.render_gt_ms(scene_name, dataset_code)
            
            if self.gt_type == "raw":
                gt_image = self.rgb_renderer.render_gt_rgb(scene_name, dataset_code)
            else:
                gt_path = os.path.join(self.dataset_root, "GT", self.gt_type + "_scenes", f"{scene_name}_{dataset_code}.png")
                if os.path.exists(gt_path):
                    gt_image = load_rgb_image(path=gt_path, bit_depth=12)
                else:
                    gt_image = self.rgb_renderer.render_gt_rgb(scene_name, dataset_code)
            
            file_name = f"{scene_name}_{dataset_code}_ILL{illuminant_idx:03d}"
        else:
            # Load from pre-generated files
            scene_name_full = file_name.split("_ILL")[0]
            ill_name = "ILL" + file_name.split("_ILL")[1]

            # Load RGB image 
            rgb_image = load_rgb_image(path=os.path.join(self.dataset_root, self.rgb_camera, "imgs", file_name+".png"), bit_depth=12)

            # Load MS image 
            ms_image, ms_wvs = read_h5(path=os.path.join(self.dataset_root, self.spectral_camera, "imgs", file_name+".h5"), key="spec")

            # Load metadata
            metadata = load_metadata(path=os.path.join(self.dataset_root, self.rgb_camera, "metadata", ill_name+".json"))
            ms_illum_spd, ill_wvs = read_h5(path=os.path.join(self.dataset_root, self.spectral_camera, "illums_spd", ill_name+".h5"), key="spec")

            assert np.all(ill_wvs == ms_wvs), f"Wavelengths of illuminant and MS camera do not match {ill_wvs}; {ms_wvs}"

            # Load MS GT
            gt_ms, _ = read_h5(path=os.path.join(self.dataset_root, self.spectral_camera, "gt_raw", scene_name_full+".h5"), key="spec")

            # Load GT IMG
            if self.gt_type == "raw":
                gt_image = load_rgb_image(os.path.join(self.dataset_root, self.rgb_camera, "gt_raw", scene_name_full+".png"), bit_depth=12)
            else:
                gt_image = load_rgb_image(path=os.path.join(self.dataset_root, "GT", self.gt_type+"_scenes", scene_name_full+".png"), bit_depth=12)

        self.rgb_means.append(np.mean(rgb_image, axis=(1,2)))
        self.rgb_stds.append(np.std(rgb_image, axis=(1,2)))
        self.ms_means.append(np.mean(ms_image, axis=(1,2)))
        self.ms_stds.append(np.std(ms_image, axis=(1,2)))

        if len(self.rgb_means) == self.__len__():
            rgb_mean = np.mean(np.array(self.rgb_means), axis=0)
            rgb_std = np.mean(np.array(self.rgb_stds), axis=0)
            ms_mean = np.mean(np.array(self.ms_means), axis=0)
            ms_std = np.mean(np.array(self.ms_stds), axis=0)
            print(f"[Dataset] Processed {self.rgb_means.__len__()} samples. RGB mean: {rgb_mean}. RGB std: {rgb_std}. MS mean: {ms_mean}. MS std: {ms_std}")

        sample = {
            "file_name": file_name,
            "rgb_image": rgb_image,
            "ms_image": ms_image,
            "ms_wvs": ms_wvs,
            "ms_illum_spd": ms_illum_spd,
            "ill_wvs": ill_wvs,
            "gt_ms": gt_ms,
            "gt_image": gt_image,
            "metadata": metadata
        }

        return sample


