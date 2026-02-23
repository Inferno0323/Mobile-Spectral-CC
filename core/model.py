import os
import importlib.util
import time
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

    def load(self, path):
        if path is not None and os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
            filtered_state_dict = {k: v for k, v in state_dict.items() 
                           if not k.endswith('total_ops') and not k.endswith('total_params')}

            self.model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Loaded model weights from {path}")
        else:
            print("No checkpoint found, initializing model with random weights")
        return self.model

    def to(self, device):
        self.model.to(device)
        self.device = device

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def parameters(self):
        return self.model.parameters()

    def num_parameters(self):
        return sum(p.numel() for p in self.model.parameters()), sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def train_step(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")

    def eval_step(self, data):
        raise NotImplementedError("This method should be overridden by subclasses")

    def train(self, mode=True):
        self.model.train(mode)
    
    def eval(self):
        self.model.eval()

    def backward_pass(self, loss):
        loss.backward()
        # If gradients go to NaN or Inf, skip the update
        if any(torch.isnan(param.grad).any() or torch.isinf(param.grad).any() for param in self.model.parameters() if param.grad is not None):
            print("NaN or Inf detected in gradients, skipping optimizer step.")
            return -1
        
        self.optimizer.step()
        return 0
    
class IlluminantEstimationModel(BaseModel):
    def __init__(self, model_name: str, model_parameters: dict):
        super(IlluminantEstimationModel, self).__init__(model_name, model_parameters)
        self.model_type = "IE"
    
        self.correction_module = ClassicCorrectionPipeline()

    def train_step(self, data):
        self.optimizer.zero_grad()

        rgb_image = data["rgb_image"].to(self.device)
        gt_image = data["gt_image"].to(self.device)
        gt_illum = data["metadata"]["illuminant_rgb"].to(self.device)
        pred_illuminant = self.model(rgb_image)

        loss = self.criterion(pred_illuminant, gt_illum)

        backward_status = self.backward_pass(loss)
        pred_xyz = self.correction_module(rgb = rgb_image, 
                                        ill = pred_illuminant.detach(), 
                                        ill_cct = data["metadata"]["ill_cct"].to(self.device), 
                                        cct_2500 = data["metadata"]["cct_2500K"].to(self.device), 
                                        cct_6500 = data["metadata"]["cct_6500K"].to(self.device))

        return loss.item(), pred_xyz, backward_status

    def eval_step(self, data):
        rgb_image = data["rgb_image"].to(self.device)
        gt_image = data["gt_image"].to(self.device)
        gt_illum = data["metadata"]["illuminant_rgb"].to(self.device)

        pred_illuminant = self.model(rgb_image)

        loss = self.criterion(pred_illuminant, gt_illum)

        pred_xyz = self.correction_module(rgb = rgb_image, 
                                        ill = pred_illuminant.detach(),
                                        ill_cct = data["metadata"]["ill_cct"].to(self.device), 
                                        cct_2500 = data["metadata"]["cct_2500K"].to(self.device), 
                                        cct_6500 = data["metadata"]["cct_6500K"].to(self.device))


        return loss.item(), pred_xyz

    def inference_step(self, rgb_image, ccts):
        rgb_image = rgb_image.to(self.device)

        pred_illuminant = self.model(rgb_image)


        pred_xyz = self.correction_module(rgb = rgb_image, 
                                        ill = pred_illuminant.detach(),
                                        ill_cct = ccts["ill_cct"].to(self.device),
                                        cct_2500 = ccts["cct_2500K"].to(self.device),
                                        cct_6500 = ccts["cct_6500K"].to(self.device)
                                        )

        return pred_xyz.clamp(0,1)

    def profile(self, n_warmup=10, n_runs=100):
        """Profile the model for FLOPs, params, inference time, and memory usage.
        
        Args:
            n_warmup: Number of warmup iterations before timing
            n_runs: Number of timed iterations for averaging
            
        Returns:
            dict with keys: flops, params, inference_time_ms, max_memory_allocated_mb, max_memory_reserved_mb
        """
        # Create dummy input
        dummy_input = torch.randn(1, 3, 512, 512).to(self.device)
        
        # Compute FLOPs and params
        flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
        
        # Inference time and memory profiling
        self.model.eval()
        
        # Reset memory stats if on CUDA
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(dummy_input)
        
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Timed runs
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = self.model(dummy_input)
        
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        avg_inference_time_ms = (end_time - start_time) / n_runs * 1000
        
        # Memory stats
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            max_memory_allocated_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            max_memory_reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        else:
            max_memory_allocated_mb = 0
            max_memory_reserved_mb = 0
        
        return {
            "flops": flops,
            "params": params,
            "inference_time_ms": avg_inference_time_ms,
            "max_memory_allocated_mb": max_memory_allocated_mb,
            "max_memory_reserved_mb": max_memory_reserved_mb
        }

class MSIlluminantEstimationModel(BaseModel):

    def __init__(self, model_name: str, model_parameters: dict):
        super(MSIlluminantEstimationModel, self).__init__(model_name, model_parameters)
        self.model_type = "MSIE"
 
        self.correction_module = ClassicCorrectionPipeline()

    def train_step(self, data):
        self.optimizer.zero_grad()

        rgb_image = data["rgb_image"].to(self.device)
        ms_image = data["ms_image"].to(self.device)
        gt_image = data["gt_image"].to(self.device)
        gt_illum = data["metadata"]["illuminant_rgb"].to(self.device)

        pred_illuminant = self.model(ms_image)

        loss = self.criterion(pred_illuminant, gt_illum)

        backward_status = self.backward_pass(loss)
        pred_xyz = self.correction_module(rgb = rgb_image, 
                                        ill = pred_illuminant.detach(), 
                                        ill_cct = data["metadata"]["ill_cct"].to(self.device), 
                                        cct_2500 = data["metadata"]["cct_2500K"].to(self.device), 
                                        cct_6500 = data["metadata"]["cct_6500K"].to(self.device))

        return loss.item(), pred_xyz, backward_status

    def eval_step(self, data):
        rgb_image = data["rgb_image"].to(self.device)
        ms_image = data["ms_image"].to(self.device)
        gt_image = data["gt_image"].to(self.device)
        gt_illum = data["metadata"]["illuminant_rgb"].to(self.device)

        pred_illuminant = self.model(ms_image)


        loss = self.criterion(pred_illuminant, gt_illum)

        pred_xyz = self.correction_module(rgb = rgb_image, 
                                        ill = pred_illuminant.detach(), 
                                        ill_cct = data["metadata"]["ill_cct"].to(self.device), 
                                        cct_2500 = data["metadata"]["cct_2500K"].to(self.device), 
                                        cct_6500 = data["metadata"]["cct_6500K"].to(self.device))


        return loss.item(), pred_xyz

    def inference_step(self, rgb_image, ms_image, ccts):
        rgb_image = rgb_image.to(self.device)
        ms_image = ms_image.to(self.device)

        pred_illuminant = self.model(ms_image)


        pred_xyz = self.correction_module(rgb = rgb_image, 
                                        ill = pred_illuminant.detach(),
                                        ill_cct = ccts["ill_cct"].to(self.device),
                                        cct_2500 = ccts["cct_2500K"].to(self.device),
                                        cct_6500 = ccts["cct_6500K"].to(self.device)
                                        )

        return pred_xyz.clamp(0,1)

    def profile(self, n_warmup=10, n_runs=100):
        """Profile the model for FLOPs, params, inference time, and memory usage.
        
        Args:
            n_warmup: Number of warmup iterations before timing
            n_runs: Number of timed iterations for averaging
            
        Returns:
            dict with keys: flops, params, inference_time_ms, max_memory_allocated_mb, max_memory_reserved_mb
        """
        # Create dummy input (MS image: 15 channels, 64x64)
        for k in spectral_channels_param.keys():
            if k.lower() in self.model_name.lower():
                param_name = spectral_channels_param[k]
                dummy_input = torch.randn(1, self.model_parameters.get(param_name, 15), 64, 64).to(self.device)
        
        # Compute FLOPs and params
        flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)

        
        # Inference time and memory profiling
        self.model.eval()
        
        # Reset memory stats if on CUDA
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(dummy_input)
        
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Timed runs
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = self.model(dummy_input)
        
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        avg_inference_time_ms = (end_time - start_time) / n_runs * 1000
        
        # Memory stats
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            max_memory_allocated_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            max_memory_reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        else:
            max_memory_allocated_mb = 0
            max_memory_reserved_mb = 0
        
        return {
            "flops": flops,
            "params": params,
            "inference_time_ms": avg_inference_time_ms,
            "max_memory_allocated_mb": max_memory_allocated_mb,
            "max_memory_reserved_mb": max_memory_reserved_mb
        }

class JointAWBModel(BaseModel):
    def __init__(self, model_name: str, model_parameters: dict):
        super(JointAWBModel, self).__init__(model_name, model_parameters)
        self.model_type = "J"
 
    def train_step(self, data):
        self.optimizer.zero_grad()

        rgb_image = data["rgb_image"].to(self.device)
        gt_image = data["gt_image"].to(self.device)
        gt_illum = data["metadata"]["illuminant_rgb"].to(self.device)

        pred_xyz = self.model(rgb_image)

        loss = self.criterion(pred_xyz, gt_image)
        backward_status = self.backward_pass(loss)
        return loss.item(), pred_xyz, backward_status

    def eval_step(self, data):
        rgb_image = data["rgb_image"].to(self.device)
        gt_image = data["gt_image"].to(self.device)
        gt_illum = data["metadata"]["illuminant_rgb"].to(self.device)

        pred_xyz = self.model(rgb_image)

        loss = self.criterion(pred_xyz, gt_image)

        return loss.item(), pred_xyz 

    def inference_step(self, rgb_image):
        rgb_image = rgb_image.to(self.device)

        pred_xyz = self.model(rgb_image)

        return pred_xyz.clamp(0,1)

    def profile(self, n_warmup=10, n_runs=100):
        """Profile the model for FLOPs, params, inference time, and memory usage.
        
        Args:
            n_warmup: Number of warmup iterations before timing
            n_runs: Number of timed iterations for averaging
            
        Returns:
            dict with keys: flops, params, inference_time_ms, max_memory_allocated_mb, max_memory_reserved_mb
        """
        # Create dummy input
        dummy_input = torch.randn(1, 3, 512, 512).to(self.device)
        
        # Compute FLOPs and params
        flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)
        
        # Inference time and memory profiling
        self.model.eval()
        
        # Reset memory stats if on CUDA
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(dummy_input)
        
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Timed runs
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = self.model(dummy_input)
        
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        avg_inference_time_ms = (end_time - start_time) / n_runs * 1000
        
        # Memory stats
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            max_memory_allocated_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            max_memory_reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        else:
            max_memory_allocated_mb = 0
            max_memory_reserved_mb = 0
        
        return {
            "flops": flops,
            "params": params,
            "inference_time_ms": avg_inference_time_ms,
            "max_memory_allocated_mb": max_memory_allocated_mb,
            "max_memory_reserved_mb": max_memory_reserved_mb
        }

class JointMSRGBAWBModel(BaseModel):
    def __init__(self, model_name: str, model_parameters: dict):
        super(JointMSRGBAWBModel, self).__init__(model_name, model_parameters)
        self.model_type = "J_MSI"

        self.criterion_proxy = AngularErrorLoss()

    def train_step(self, data):
        self.optimizer.zero_grad()


        rgb_image = data["rgb_image"].to(self.device)
        ms_image = data["ms_image"].to(self.device)
        gt_image = data["gt_image"].to(self.device)
        gt_illum = data["metadata"]["illuminant_rgb"].to(self.device)

        pred_xyz = self.model(rgb_image, ms_image)

        loss = self.criterion(pred_xyz, gt_image)
        
        backward_status = self.backward_pass(loss)
        return loss.item(), pred_xyz, backward_status 

    def eval_step(self, data):
        rgb_image = data["rgb_image"].to(self.device)
        ms_image = data["ms_image"].to(self.device)
        gt_image = data["gt_image"].to(self.device)
        gt_illum = data["metadata"]["illuminant_rgb"].to(self.device)

        pred_xyz = self.model(rgb_image, ms_image)

        loss = self.criterion(pred_xyz, gt_image)

        return loss.item(), pred_xyz 

    def inference_step(self, rgb_image, ms_image):
        rgb_image = rgb_image.to(self.device)
        ms_image = ms_image.to(self.device)

        pred_xyz = self.model(rgb_image, ms_image)

        return pred_xyz.clamp(0,1)

    def profile(self, n_warmup=10, n_runs=100):
        """Profile the model for FLOPs, params, inference time, and memory usage.
        
        Args:
            n_warmup: Number of warmup iterations before timing
            n_runs: Number of timed iterations for averaging
            
        Returns:
            dict with keys: flops, params, inference_time_ms, max_memory_allocated_mb, max_memory_reserved_mb
        """
        # Create dummy inputs (RGB: 3 channels 512x512, MSI: 15 channels 64x64)
        dummy_rgb = torch.randn(1, 3, 512, 512).to(self.device)
        
        for k in spectral_channels_param.keys():
            if k.lower() in self.model_name.lower():
                param_name = spectral_channels_param[k]
                dummy_msi = torch.randn(1, self.model_parameters.get(param_name, 15), 64, 64).to(self.device)
        
        # Compute FLOPs and params
        flops, params = profile(self.model, inputs=(dummy_rgb, dummy_msi), verbose=False)
        
        # Inference time and memory profiling
        self.model.eval()
        
        # Reset memory stats if on CUDA
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(dummy_rgb, dummy_msi)
        
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Timed runs
        start_time = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = self.model(dummy_rgb, dummy_msi)
        
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        avg_inference_time_ms = (end_time - start_time) / n_runs * 1000
        
        # Memory stats
        if self.device != torch.device("cpu") and torch.cuda.is_available():
            max_memory_allocated_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            max_memory_reserved_mb = torch.cuda.max_memory_reserved(self.device) / (1024 ** 2)
        else:
            max_memory_allocated_mb = 0
            max_memory_reserved_mb = 0
        
        return {
            "flops": flops,
            "params": params,
            "inference_time_ms": avg_inference_time_ms,
            "max_memory_allocated_mb": max_memory_allocated_mb,
            "max_memory_reserved_mb": max_memory_reserved_mb
        }
