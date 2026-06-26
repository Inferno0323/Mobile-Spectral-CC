import os
import importlib.util
import time
import copy
from contextlib import nullcontext
from auxiliary.metrics import *
from models import *
from thop import profile
import torch 


spectral_channels_param = {
    "cmKAN" :'in_spec',
    "ConvMean": 'inp_size',
    "FC4": 'inp_size',
    "LPIENet": 'spectral_input_channels',
}

class BaseModel(torch.nn.Module):
    def __init__(self, model_name: str, model_parameters: dict):
        super(BaseModel, self).__init__()
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.model = self.initialize_model()
        self.device = torch.device("cpu")
        self.device_ids = None
        self.amp_enabled = False
        self.amp_dtype = torch.float16
        self.scaler = None
        self.channels_last = False
        self.non_blocking = False
        self.check_gradients = True

    def initialize_model(self):
        if os.path.exists(self.model_name): # model_name is a path
            # Import the model from the given path
            spec = importlib.util.spec_from_file_location("custom_model", self.model_name)
            custom_model = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_model)
            model = custom_model.CustomModel(**self.model_parameters)

        else: # model_name is a predefined model
            model = eval(self.model_name)(**self.model_parameters)

        return model

    def init_optimizer(self, lr):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return self.optimizer

    def init_loss_criterion(self, criterion):
        self.criterion = eval(criterion)()

    def configure_performance(self, amp=False, amp_dtype="float16", channels_last=False, non_blocking=False, check_gradients=True):
        self.amp_enabled = bool(amp) and self._using_cuda()
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        self.channels_last = bool(channels_last)
        self.non_blocking = bool(non_blocking) and self._using_cuda()
        self.check_gradients = bool(check_gradients)
        self.scaler = self._make_grad_scaler()

        if self.channels_last and self._using_cuda():
            self.model.to(memory_format=torch.channels_last)

    def _make_grad_scaler(self):
        if not self.amp_enabled:
            return None
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler(self.device.type, enabled=True)
        return torch.cuda.amp.GradScaler(enabled=True)

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=True)
        return torch.cuda.amp.autocast(dtype=self.amp_dtype, enabled=True)

    def _move(self, tensor):
        tensor = tensor.to(self.device, non_blocking=self.non_blocking)
        if self.channels_last and tensor.ndim == 4 and self._using_cuda():
            tensor = tensor.contiguous(memory_format=torch.channels_last)
        return tensor

    def _metric_tensor(self, tensor):
        return tensor.float() if self.amp_enabled else tensor

    def _raw_model(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model

    @staticmethod
    def _normalize_state_dict(state_dict):
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k.endswith('total_ops') or k.endswith('total_params'):
                continue

            key = k
            for prefix in ("model.module.", "model.", "module."):
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            filtered_state_dict[key] = v

        return filtered_state_dict

    def load(self, path):
        if path is not None and os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
            filtered_state_dict = self._normalize_state_dict(state_dict)

            self._raw_model().load_state_dict(filtered_state_dict, strict=False)
            print(f"Loaded model weights from {path}")
        else:
            print("No checkpoint found, initializing model with random weights")
        return self.model

    def _validate_device_ids(self, device_ids):
        if device_ids is None:
            return None

        device_ids = [int(device_id) for device_id in device_ids]
        if len(device_ids) <= 1:
            return None

        if not torch.cuda.is_available():
            raise RuntimeError("DataParallel requested, but CUDA is not available.")

        device_count = torch.cuda.device_count()
        invalid_device_ids = [device_id for device_id in device_ids if device_id < 0 or device_id >= device_count]
        if invalid_device_ids:
            raise ValueError(
                f"Invalid CUDA device id(s) {invalid_device_ids}; available device ids are 0..{device_count - 1}."
            )

        return device_ids

    def to(self, device, device_ids=None):
        device = torch.device(device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA device {device} requested, but CUDA is not available.")
            if device.index is None:
                device = torch.device("cuda:0")

        model = self._raw_model()
        device_ids = self._validate_device_ids(device_ids)

        if device.type == "cuda" and device_ids is not None:
            if device.index not in device_ids:
                device_ids = [device.index] + device_ids
            elif device_ids[0] != device.index:
                device_ids = [device.index] + [device_id for device_id in device_ids if device_id != device.index]

            model.to(device)
            self.model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
            self.device = torch.device(f"cuda:{device_ids[0]}")
            self.device_ids = device_ids
            return self

        self.model = model.to(device)
        self.device = device
        self.device_ids = None
        return self

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def state_dict(self, *args, **kwargs):
        return self._raw_model().state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self._raw_model().load_state_dict(self._normalize_state_dict(state_dict), strict=strict)

    def parameters(self):
        return self.model.parameters()

    def num_parameters(self):
        model = self._raw_model()
        return sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train_step(self, data, compute_metrics=True):
        raise NotImplementedError("This method should be overridden by subclasses")

    def eval_step(self, data, compute_metrics=True):
        raise NotImplementedError("This method should be overridden by subclasses")

    def train(self, mode=True):
        self.model.train(mode)
        return self
    
    def eval(self):
        self.model.eval()
        return self

    def backward_pass(self, loss):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()

        # If gradients go to NaN or Inf, skip the update
        if self.check_gradients and any(torch.isnan(param.grad).any() or torch.isinf(param.grad).any() for param in self.model.parameters() if param.grad is not None):
            print("NaN or Inf detected in gradients, skipping optimizer step.")
            if self.scaler is not None:
                self.scaler.update()
            return -1

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        return 0

    def _using_cuda(self):
        return self.device.type == "cuda" and torch.cuda.is_available()

    def _synchronize_device(self):
        if self._using_cuda():
            torch.cuda.synchronize(self.device)

    def _profile(self, inputs, n_warmup, n_runs):
        inputs = tuple(input_tensor.to(self.device) for input_tensor in inputs)
        was_training = self.model.training
        self.model.eval()

        # THOP mutates modules by adding profiling buffers. Run it on a copy so
        # DataParallel never sees CPU-side total_ops/total_params buffers.
        profile_model = copy.deepcopy(self._raw_model()).to(self.device)
        profile_model.eval()
        flops, params = profile(profile_model, inputs=inputs, verbose=False)
        del profile_model

        if self._using_cuda():
            torch.cuda.reset_peak_memory_stats(self.device)
            self._synchronize_device()

        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(*inputs)

        self._synchronize_device()

        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = self.model(*inputs)

        self._synchronize_device()

        end_time = time.perf_counter()
        avg_inference_time_ms = (end_time - start_time) / n_runs * 1000

        if self._using_cuda():
            max_memory_allocated_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            max_memory_reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        else:
            max_memory_allocated_mb = 0
            max_memory_reserved_mb = 0

        self.model.train(was_training)

        return {
            "flops": flops,
            "params": params,
            "inference_time_ms": avg_inference_time_ms,
            "max_memory_allocated_mb": max_memory_allocated_mb,
            "max_memory_reserved_mb": max_memory_reserved_mb
        }

    def _spectral_input_channels(self, default=15):
        for k in spectral_channels_param.keys():
            if k.lower() in self.model_name.lower():
                param_name = spectral_channels_param[k]
                return self.model_parameters.get(param_name, default)
        return default
    
class IlluminantEstimationModel(BaseModel):
    def __init__(self, model_name: str, model_parameters: dict):
        super(IlluminantEstimationModel, self).__init__(model_name, model_parameters)
        self.model_type = "IE"
    
        self.correction_module = ClassicCorrectionPipeline()

    def train_step(self, data, compute_metrics=True):
        self.optimizer.zero_grad(set_to_none=True)

        rgb_image = self._move(data["rgb_image"])
        gt_illum = self._move(data["metadata"]["illuminant_rgb"])

        with self._autocast():
            pred_illuminant = self.model(rgb_image)
            loss = self.criterion(pred_illuminant, gt_illum)

        backward_status = self.backward_pass(loss)
        if not compute_metrics:
            return loss.item(), None, backward_status

        pred_xyz = self.correction_module(rgb = rgb_image, 
                                        ill = self._metric_tensor(pred_illuminant.detach()),
                                        ill_cct = self._move(data["metadata"]["ill_cct"]),
                                        cct_2500 = self._move(data["metadata"]["cct_2500K"]),
                                        cct_6500 = self._move(data["metadata"]["cct_6500K"]))

        return loss.item(), self._metric_tensor(pred_xyz), backward_status

    def eval_step(self, data, compute_metrics=True):
        rgb_image = self._move(data["rgb_image"])
        gt_illum = self._move(data["metadata"]["illuminant_rgb"])

        with self._autocast():
            pred_illuminant = self.model(rgb_image)
            loss = self.criterion(pred_illuminant, gt_illum)

        if not compute_metrics:
            return loss.item(), None

        pred_xyz = self.correction_module(rgb = rgb_image, 
                                        ill = self._metric_tensor(pred_illuminant.detach()),
                                        ill_cct = self._move(data["metadata"]["ill_cct"]),
                                        cct_2500 = self._move(data["metadata"]["cct_2500K"]),
                                        cct_6500 = self._move(data["metadata"]["cct_6500K"]))


        return loss.item(), self._metric_tensor(pred_xyz)

    def inference_step(self, rgb_image, ccts):
        rgb_image = self._move(rgb_image)

        with self._autocast():
            pred_illuminant = self.model(rgb_image)


        pred_xyz = self.correction_module(rgb = rgb_image, 
                                        ill = self._metric_tensor(pred_illuminant.detach()),
                                        ill_cct = self._move(ccts["ill_cct"]),
                                        cct_2500 = self._move(ccts["cct_2500K"]),
                                        cct_6500 = self._move(ccts["cct_6500K"])
                                        )

        return self._metric_tensor(pred_xyz).clamp(0,1)

    def profile(self, n_warmup=10, n_runs=100):
        """Profile the model for FLOPs, params, inference time, and memory usage.
        
        Args:
            n_warmup: Number of warmup iterations before timing
            n_runs: Number of timed iterations for averaging
            
        Returns:
            dict with keys: flops, params, inference_time_ms, max_memory_allocated_mb, max_memory_reserved_mb
        """
        dummy_input = torch.randn(1, 3, 512, 512, device=self.device)
        return self._profile((dummy_input,), n_warmup, n_runs)

class MSIlluminantEstimationModel(BaseModel):

    def __init__(self, model_name: str, model_parameters: dict):
        super(MSIlluminantEstimationModel, self).__init__(model_name, model_parameters)
        self.model_type = "MSIE"
 
        self.correction_module = ClassicCorrectionPipeline()

    def train_step(self, data, compute_metrics=True):
        self.optimizer.zero_grad(set_to_none=True)

        rgb_image = self._move(data["rgb_image"])
        ms_image = self._move(data["ms_image"])
        gt_illum = self._move(data["metadata"]["illuminant_rgb"])

        with self._autocast():
            pred_illuminant = self.model(ms_image)
            loss = self.criterion(pred_illuminant, gt_illum)

        backward_status = self.backward_pass(loss)
        if not compute_metrics:
            return loss.item(), None, backward_status

        pred_xyz = self.correction_module(rgb = rgb_image, 
                                        ill = self._metric_tensor(pred_illuminant.detach()),
                                        ill_cct = self._move(data["metadata"]["ill_cct"]),
                                        cct_2500 = self._move(data["metadata"]["cct_2500K"]),
                                        cct_6500 = self._move(data["metadata"]["cct_6500K"]))

        return loss.item(), self._metric_tensor(pred_xyz), backward_status

    def eval_step(self, data, compute_metrics=True):
        rgb_image = self._move(data["rgb_image"])
        ms_image = self._move(data["ms_image"])
        gt_illum = self._move(data["metadata"]["illuminant_rgb"])


        with self._autocast():
            pred_illuminant = self.model(ms_image)
            loss = self.criterion(pred_illuminant, gt_illum)

        if not compute_metrics:
            return loss.item(), None

        pred_xyz = self.correction_module(rgb = rgb_image, 
                                        ill = self._metric_tensor(pred_illuminant.detach()),
                                        ill_cct = self._move(data["metadata"]["ill_cct"]),
                                        cct_2500 = self._move(data["metadata"]["cct_2500K"]),
                                        cct_6500 = self._move(data["metadata"]["cct_6500K"]))


        return loss.item(), self._metric_tensor(pred_xyz)

    def inference_step(self, rgb_image, ms_image, ccts):
        rgb_image = self._move(rgb_image)
        ms_image = self._move(ms_image)

        with self._autocast():
            pred_illuminant = self.model(ms_image)


        pred_xyz = self.correction_module(rgb = rgb_image, 
                                        ill = self._metric_tensor(pred_illuminant.detach()),
                                        ill_cct = self._move(ccts["ill_cct"]),
                                        cct_2500 = self._move(ccts["cct_2500K"]),
                                        cct_6500 = self._move(ccts["cct_6500K"])
                                        )

        return self._metric_tensor(pred_xyz).clamp(0,1)

    def profile(self, n_warmup=10, n_runs=100):
        """Profile the model for FLOPs, params, inference time, and memory usage.
        
        Args:
            n_warmup: Number of warmup iterations before timing
            n_runs: Number of timed iterations for averaging
            
        Returns:
            dict with keys: flops, params, inference_time_ms, max_memory_allocated_mb, max_memory_reserved_mb
        """
        dummy_input = torch.randn(1, self._spectral_input_channels(), 64, 64, device=self.device)
        return self._profile((dummy_input,), n_warmup, n_runs)

class JointAWBModel(BaseModel):
    def __init__(self, model_name: str, model_parameters: dict):
        super(JointAWBModel, self).__init__(model_name, model_parameters)
        self.model_type = "J"
 
    def train_step(self, data, compute_metrics=True):
        self.optimizer.zero_grad(set_to_none=True)

        rgb_image = self._move(data["rgb_image"])
        gt_image = self._move(data["gt_image"])

        with self._autocast():
            pred_xyz = self.model(rgb_image)
            loss = self.criterion(pred_xyz, gt_image)
        backward_status = self.backward_pass(loss)
        if not compute_metrics:
            return loss.item(), None, backward_status

        return loss.item(), self._metric_tensor(pred_xyz), backward_status

    def eval_step(self, data, compute_metrics=True):
        rgb_image = self._move(data["rgb_image"])
        gt_image = self._move(data["gt_image"])

        with self._autocast():
            pred_xyz = self.model(rgb_image)
            loss = self.criterion(pred_xyz, gt_image)

        if not compute_metrics:
            return loss.item(), None

        return loss.item(), self._metric_tensor(pred_xyz)

    def inference_step(self, rgb_image):
        rgb_image = self._move(rgb_image)

        with self._autocast():
            pred_xyz = self.model(rgb_image)

        return self._metric_tensor(pred_xyz).clamp(0,1)

    def profile(self, n_warmup=10, n_runs=100):
        """Profile the model for FLOPs, params, inference time, and memory usage.
        
        Args:
            n_warmup: Number of warmup iterations before timing
            n_runs: Number of timed iterations for averaging
            
        Returns:
            dict with keys: flops, params, inference_time_ms, max_memory_allocated_mb, max_memory_reserved_mb
        """
        dummy_input = torch.randn(1, 3, 512, 512, device=self.device)
        return self._profile((dummy_input,), n_warmup, n_runs)

class JointMSRGBAWBModel(BaseModel):
    def __init__(self, model_name: str, model_parameters: dict):
        super(JointMSRGBAWBModel, self).__init__(model_name, model_parameters)
        self.model_type = "J_MSI"

        self.criterion_proxy = AngularErrorLoss()

    def train_step(self, data, compute_metrics=True):
        self.optimizer.zero_grad(set_to_none=True)


        rgb_image = self._move(data["rgb_image"])
        ms_image = self._move(data["ms_image"])
        gt_image = self._move(data["gt_image"])

        with self._autocast():
            pred_xyz = self.model(rgb_image, ms_image)
            loss = self.criterion(pred_xyz, gt_image)
        
        backward_status = self.backward_pass(loss)
        if not compute_metrics:
            return loss.item(), None, backward_status

        return loss.item(), self._metric_tensor(pred_xyz), backward_status

    def eval_step(self, data, compute_metrics=True):
        rgb_image = self._move(data["rgb_image"])
        ms_image = self._move(data["ms_image"])
        gt_image = self._move(data["gt_image"])

        with self._autocast():
            pred_xyz = self.model(rgb_image, ms_image)
            loss = self.criterion(pred_xyz, gt_image)

        if not compute_metrics:
            return loss.item(), None

        return loss.item(), self._metric_tensor(pred_xyz)

    def inference_step(self, rgb_image, ms_image):
        rgb_image = self._move(rgb_image)
        ms_image = self._move(ms_image)

        with self._autocast():
            pred_xyz = self.model(rgb_image, ms_image)

        return self._metric_tensor(pred_xyz).clamp(0,1)

    def profile(self, n_warmup=10, n_runs=100):
        """Profile the model for FLOPs, params, inference time, and memory usage.
        
        Args:
            n_warmup: Number of warmup iterations before timing
            n_runs: Number of timed iterations for averaging
            
        Returns:
            dict with keys: flops, params, inference_time_ms, max_memory_allocated_mb, max_memory_reserved_mb
        """
        dummy_rgb = torch.randn(1, 3, 512, 512, device=self.device)
        dummy_msi = torch.randn(1, self._spectral_input_channels(), 64, 64, device=self.device)
        return self._profile((dummy_rgb, dummy_msi), n_warmup, n_runs)


class RGBSpectralPriorModel(BaseModel):
    def __init__(self, model_name: str, model_parameters: dict):
        model_parameters = dict(model_parameters)
        self.spectral_loss_weight = model_parameters.pop("spectral_loss_weight", 0.1)
        super(RGBSpectralPriorModel, self).__init__(model_name, model_parameters)
        self.model_type = "RGB_SP"
        self.spectral_criterion = torch.nn.L1Loss()

    def train_step(self, data, compute_metrics=True):
        self.optimizer.zero_grad(set_to_none=True)

        rgb_image = self._move(data["rgb_image"])
        ms_image = self._move(data["ms_image"])
        gt_image = self._move(data["gt_image"])

        with self._autocast():
            pred_rgb, pred_ms = self.model(rgb_image)
            image_loss = self.criterion(pred_rgb, gt_image)
            ms_target = ms_image
            if ms_target.shape[-2:] != pred_ms.shape[-2:]:
                ms_target = torch.nn.functional.interpolate(ms_target, size=pred_ms.shape[-2:], mode="bilinear", align_corners=False)
            spectral_loss = self.spectral_criterion(pred_ms, ms_target)
            loss = image_loss + self.spectral_loss_weight * spectral_loss

        backward_status = self.backward_pass(loss)
        if not compute_metrics:
            return loss.item(), None, backward_status

        return loss.item(), self._metric_tensor(pred_rgb), backward_status

    def eval_step(self, data, compute_metrics=True):
        rgb_image = self._move(data["rgb_image"])
        gt_image = self._move(data["gt_image"])

        with self._autocast():
            pred_rgb, _ = self.model(rgb_image)
            loss = self.criterion(pred_rgb, gt_image)

        if not compute_metrics:
            return loss.item(), None

        return loss.item(), self._metric_tensor(pred_rgb)

    def inference_step(self, rgb_image):
        rgb_image = self._move(rgb_image)
        with self._autocast():
            pred_rgb, _ = self.model(rgb_image)
        return self._metric_tensor(pred_rgb).clamp(0, 1)

    def profile(self, n_warmup=10, n_runs=100):
        dummy_rgb = torch.randn(1, 3, 512, 512, device=self.device)
        return self._profile((dummy_rgb,), n_warmup, n_runs)
