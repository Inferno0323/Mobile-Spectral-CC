import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from datetime import datetime
from auxiliary.experiment_utils import seed_everything
from torch.utils.data import DataLoader
import shutil
import numpy as np
import tqdm
from datasets import RGBDataset, MSDataset, RGBMSDataset
from core.model import IlluminantEstimationModel, JointAWBModel, MSIlluminantEstimationModel, JointMSRGBAWBModel, RGBSpectralPriorModel
from core.evaluator import Evaluator
from core.logger import Logger
from auxiliary.metrics import *

import ipdb

class Experiment():

    def __init__(self, cfg=None, filepath=None, **kwargs):
        if cfg is not None:
            self.from_dict(cfg)
        elif filepath is not None:
            cfg = self.from_file(filepath)
            self.from_dict(cfg)
        else:
            raise ValueError("Either cfg or filepath must be provided")
        
        # Override config parameters with kwargs
        for k, v in kwargs.items():
            if k == "model_parameter_overrides":
                self.model_parameters.update(v)
                self.cfg["model_parameters"] = self.model_parameters
            elif k == "train_metrics":
                self.train_metrics_enabled = v
                self.cfg[k] = v
            elif k == "val_metrics":
                self.val_metrics_enabled = v
                self.cfg[k] = v
            else:
                setattr(self, k, v)
                self.cfg[k] = v

        self.prepare_directory()

        seed_everything(self.seed, deterministic=self.deterministic)
        self.configure_torch_runtime()

        self.prepare_data() 
        self.prepare_model()
        if self.train:
            self.prepare_training()
        if self.test:
            self.prepare_test()

        self.logger = Logger(self.to_dict(), self.exp_dir)
        
    def from_file(self, filepath):
        config = {}
        with open(filepath, "r") as f:
            exec(f.read(), config)
        return config["cfg"]
    
    def from_dict(self, cfg):
        self.cfg = cfg

        self.exp_name = cfg["exp_name"]
        self.train = cfg["train"]
        self.test = cfg["test"]
        self.model_name = cfg["model_name"]
        self.model_type = cfg["model_type"]
        self.model_parameters = cfg.get("model_parameters", dict())

        self.data_type = cfg["data_type"]
        self.dataset_root = cfg["dataset_root"]
        if self.data_type in ["RGB", "RGB+MS"]:
            self.rgb_camera = cfg.get("rgb_camera", None)
        self.gt_type = cfg["gt_type"]
        if self.data_type in ["MS", "RGB+MS"]:
            self.spectral_camera = cfg.get("spectral_camera", None)
        self.repeat = cfg.get("repeat", 1)
        self.train_list = cfg["train_list"]
        self.val_list = cfg["val_list"]
        self.test_list = cfg["test_list"]
        self.val_viz_list = cfg["val_viz_list"]
        self.test_viz_list = cfg["test_viz_list"]
        
        # Optional: deltaE00 range for saving test visualizations [min, max]
        # If specified, images with deltaE00 in this range will also be saved
        self.test_viz_de00_range = cfg.get("test_viz_de00_range", None)
        
        # Optional: misaligned MS images
        self.misaligned = cfg.get("misaligned", False)

        if self.train:
            self.n_epochs = cfg["n_epochs"]
            self.lr = cfg["lr"]
            self.train_batch_size = cfg["train_batch_size"]
            self.val_batch_size = cfg["val_batch_size"]
            self.early_stop = cfg["early_stop"]
        self.seed = cfg["seed"]
        self.criterion = cfg["criterion"]
        self.metrics = cfg.get("metrics", ["deltaE00"])
        self.device = cfg["device"]
        self.device_ids = None
        self.data_parallel = cfg.get("data_parallel", None)
        self.configured_device_ids = cfg.get("device_ids", None)
        self.n_workers = cfg["n_workers"]
        self.test_batch_size = cfg["test_batch_size"]
        self.exp_dir = cfg["exp_dir"]
        self.train_checkpoint = cfg.get("train_checkpoint", None)
        self.pretrained_weights = cfg.get("pretrained_weights", None)
        self.deterministic = cfg.get("deterministic", True)
        self.amp = cfg.get("amp", False)
        self.amp_dtype = cfg.get("amp_dtype", "float16")
        self.tf32 = cfg.get("tf32", False)
        self.channels_last = cfg.get("channels_last", False)
        self.non_blocking = cfg.get("non_blocking", False)
        self.profile_model = cfg.get("profile_model", True)
        self.persistent_workers = cfg.get("persistent_workers", False)
        self.prefetch_factor = cfg.get("prefetch_factor", None)
        self.train_metrics_enabled = cfg.get("train_metrics", True)
        self.val_metrics_enabled = cfg.get("val_metrics", True)
        self.val_interval = cfg.get("val_interval", 1)
        self.cache_rgb = cfg.get("cache_rgb", False)
        self.cache_dir = cfg.get("cache_dir", None)
        self.check_gradients = cfg.get("check_gradients", True)
        self.plot_metrics_enabled = cfg.get("plot_metrics", True)
        self.tensorboard = cfg.get("tensorboard", False)
        self.tensorboard_images = cfg.get("tensorboard_images", True)
        self.tensorboard_image_interval = cfg.get("tensorboard_image_interval", 5)

    def to_dict(self):
        return self.cfg

    def configure_torch_runtime(self):
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self.tf32
            torch.backends.cudnn.allow_tf32 = self.tf32
            if self.tf32 and hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

    @staticmethod
    def resolve_device_config(device_config, configured_device_ids=None, data_parallel=None):
        def parse_device_ids(device_ids):
            if isinstance(device_ids, str):
                normalized_device_ids = device_ids.strip().lower()
                if normalized_device_ids in ("all", "cuda"):
                    return list(range(torch.cuda.device_count()))
                return [int(device_id.strip()) for device_id in normalized_device_ids.split(",") if device_id.strip()]
            return [int(device_id) for device_id in device_ids]

        if isinstance(device_config, str):
            normalized_device_config = device_config.strip().lower()
            if normalized_device_config in ("cpu", "-1"):
                device_config = -1
            elif normalized_device_config in ("all", "cuda"):
                device_config = "all"
            elif "," in normalized_device_config:
                device_config = parse_device_ids(normalized_device_config)
            else:
                device_config = int(normalized_device_config)

        if isinstance(device_config, (list, tuple)):
            if len(device_config) == 0 or int(device_config[0]) < 0:
                return torch.device("cpu"), None
            device_ids = [int(device_id) for device_id in device_config]
        elif device_config == "all":
            if not torch.cuda.is_available():
                raise RuntimeError("All CUDA devices requested, but CUDA is not available.")
            device_ids = list(range(torch.cuda.device_count()))
        elif int(device_config) < 0:
            return torch.device("cpu"), None
        else:
            primary_device_id = int(device_config)
            device_ids = [primary_device_id]

        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device(s) {device_ids} requested, but CUDA is not available.")

        device_count = torch.cuda.device_count()
        invalid_device_ids = [device_id for device_id in device_ids if device_id < 0 or device_id >= device_count]
        if invalid_device_ids:
            raise ValueError(
                f"Invalid CUDA device id(s) {invalid_device_ids}; available device ids are 0..{device_count - 1}."
            )

        if configured_device_ids is not None:
            device_ids = parse_device_ids(configured_device_ids)
        elif data_parallel is True or (data_parallel is None and len(device_ids) == 1 and device_count > 1):
            primary_device_id = device_ids[0]
            device_ids = [primary_device_id] + [device_id for device_id in range(device_count) if device_id != primary_device_id]

        invalid_device_ids = [device_id for device_id in device_ids if device_id < 0 or device_id >= device_count]
        if invalid_device_ids:
            raise ValueError(
                f"Invalid CUDA device id(s) {invalid_device_ids}; available device ids are 0..{device_count - 1}."
            )

        device_ids = list(dict.fromkeys(device_ids))
        if len(device_ids) == 0:
            raise ValueError("At least one CUDA device id must be provided.")

        primary_device_id = device_ids[0]
        parallel_device_ids = device_ids if len(device_ids) > 1 else None
        return torch.device(f"cuda:{primary_device_id}"), parallel_device_ids
    
    def prepare_directory(self):
        if self.exp_dir is None:
            self.exp_dir = os.path.join("./experiments/", datetime.now().strftime("%y%m%d_%H%M%S") + "_" + self.exp_name)

        self.device, self.device_ids = self.resolve_device_config(
            self.device,
            configured_device_ids=self.configured_device_ids,
            data_parallel=self.data_parallel,
        )
        os.makedirs(self.exp_dir, exist_ok=True)
        if self.val_viz_list:
            os.makedirs(os.path.join(self.exp_dir, "val_viz"), exist_ok=True)
        if self.test_viz_list or self.test_viz_de00_range:
            os.makedirs(os.path.join(self.exp_dir, "test_viz"), exist_ok=True)

        # Save a copy of the config file in the experiment directory
        with open(os.path.join(self.exp_dir, "config.py"), "w") as f:
            f.write("cfg = dict(\n")
            for k, v in self.cfg.items():
                f.write(f"{k} = {repr(v)},\n")
            f.write(")\n")

        # If resuming from checkpoint, copy the checkpoint file to the experiment directory
        if self.train_checkpoint is not None:
            try:
                shutil.copyfile(self.train_checkpoint, os.path.join(self.exp_dir, os.path.basename(self.train_checkpoint)), follow_symlinks=True)
            except shutil.SameFileError:
                pass
        if self.pretrained_weights is not None:
            try:
                shutil.copyfile(self.pretrained_weights, os.path.join(self.exp_dir, os.path.basename(self.pretrained_weights)), follow_symlinks=True)
            except shutil.SameFileError:
                pass

    def prepare_data(self):
        pin_memory = self.device.type == "cuda"
        train_load_gt = self.train_metrics_enabled or self.model_type != "IE"
        val_load_gt = self.val_metrics_enabled or self.model_type != "IE"
        train_input_size = None if train_load_gt else self.model_parameters.get("input_size", None)
        val_input_size = None if val_load_gt else self.model_parameters.get("input_size", None)
        if self.data_type == "RGB":
            self.train_dataset = RGBDataset(self.dataset_root, self.train_list, self.rgb_camera, self.gt_type,
                                            is_train=True, seed=self.seed, load_gt=train_load_gt, input_size=train_input_size,
                                            cache_rgb=self.cache_rgb and not train_load_gt, cache_dir=self.cache_dir)
            self.val_dataset = RGBDataset(self.dataset_root, self.val_list, self.rgb_camera, self.gt_type,
                                          is_train=False, seed=self.seed, load_gt=val_load_gt, input_size=val_input_size,
                                          cache_rgb=self.cache_rgb and not val_load_gt, cache_dir=self.cache_dir)
            self.test_dataset = RGBDataset(self.dataset_root, self.test_list, self.rgb_camera, self.gt_type,
                                           is_train=False, seed=self.seed)
        elif self.data_type == "MS":
            self.train_dataset = MSDataset(self.dataset_root, self.train_list, self.spectral_camera, self.gt_type,
                                           is_train=True, seed=self.seed, misaligned=self.misaligned)
            self.val_dataset = MSDataset(self.dataset_root, self.val_list, self.spectral_camera, self.gt_type,
                                         is_train=False, seed=self.seed, misaligned=self.misaligned)
            self.test_dataset = MSDataset(self.dataset_root, self.test_list, self.spectral_camera, self.gt_type,
                                          is_train=False, seed=self.seed, misaligned=self.misaligned)
        elif self.data_type == "RGB+MS":
            self.train_dataset = RGBMSDataset(self.dataset_root, self.train_list, self.rgb_camera, self.spectral_camera, self.gt_type,
                                              is_train=True, seed=self.seed, misaligned=self.misaligned)
            self.val_dataset = RGBMSDataset(self.dataset_root, self.val_list, self.rgb_camera, self.spectral_camera, self.gt_type,
                                            is_train=False, seed=self.seed, misaligned=self.misaligned)
            self.test_dataset = RGBMSDataset(self.dataset_root, self.test_list, self.rgb_camera, self.spectral_camera, self.gt_type,
                                             is_train=False, seed=self.seed, misaligned=self.misaligned)
        
        g = torch.Generator()
        g.manual_seed(self.seed)
        loader_kwargs = dict(
            num_workers=self.n_workers,
            pin_memory=pin_memory,
            persistent_workers=self.persistent_workers and self.n_workers > 0,
        )
        if self.prefetch_factor is not None and self.n_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        if self.train:
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, generator=g, **loader_kwargs)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, **loader_kwargs)
        if self.test:
            self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, **loader_kwargs)

    def prepare_model(self):
        if self.model_type == "IE":
            self.model = IlluminantEstimationModel(self.model_name, self.model_parameters)
        elif self.model_type == "J":
            self.model = JointAWBModel(self.model_name, self.model_parameters)
        elif self.model_type == "MSIE":
            self.model = MSIlluminantEstimationModel(self.model_name, self.model_parameters)
        elif self.model_type == "J_MSI":
            self.model = JointMSRGBAWBModel(self.model_name, self.model_parameters)
        elif self.model_type == "RGB_SP":
            self.model = RGBSpectralPriorModel(self.model_name, self.model_parameters)

    def prepare_training(self):
        self.prepare_test()
        self.model.init_optimizer(self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.model.optimizer, T_max=self.n_epochs, eta_min=1e-6)
        if self.train_checkpoint is not None:
            self.load_checkpoint(self.train_checkpoint)
        else:
            if self.pretrained_weights is not None:
                self.model.load(self.pretrained_weights)
            self.train_metrics = Evaluator(self.metrics if self.train_metrics_enabled else [])
            self.val_metrics = Evaluator(self.metrics if self.val_metrics_enabled else [])
            self.best_loss = np.inf
            self.early_stop_counter = 0
            self.starting_epoch = 0

    def prepare_test(self):
        self.model.to(self.device, device_ids=self.device_ids)
        self.model.configure_performance(
            amp=self.amp,
            amp_dtype=self.amp_dtype,
            channels_last=self.channels_last,
            non_blocking=self.non_blocking,
            check_gradients=self.check_gradients,
        )
        self.model.init_loss_criterion(self.criterion)
        if self.pretrained_weights is not None:
            self.model.load(self.pretrained_weights)
        self.test_metrics = Evaluator(self.metrics)

    def save_checkpoint(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.model.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_loss': self.best_loss,
            'early_stop_counter': self.early_stop_counter,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        filtered_state_dict = {k: v for k, v in ckpt['model_state_dict'].items() 
                           if not k.endswith('total_ops') and not k.endswith('total_params')}
        self.model.load_state_dict(filtered_state_dict, strict=False)
        self.model.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.train_metrics = ckpt['train_metrics']
        self.val_metrics = ckpt['val_metrics']
        self.best_loss = ckpt['best_loss']
        self.early_stop_counter = ckpt['early_stop_counter']
        self.starting_epoch = ckpt['epoch']+1


