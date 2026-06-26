import argparse
from core.runner import Runner
import ipdb


def parse_device_arg(device):
    normalized_device = device.strip().lower()
    if normalized_device in ("cpu", "-1"):
        return -1
    if normalized_device in ("all", "cuda"):
        return "all"
    if "," in normalized_device:
        return [int(device_id.strip()) for device_id in normalized_device.split(",") if device_id.strip()]
    return int(normalized_device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Specify the config file to be used")

    # Optional arguments to override config parameters
    parser.add_argument("--device", type=str, default=None, help="Specify the GPU id(s) to use (-1/cpu, 0, 0,1, or all)")
    parser.add_argument("--workers", type=int, default=None, help="Specify the number of workers")
    parser.add_argument("--fast", action="store_true", help="Enable high-throughput GPU settings: AMP, TF32, cuDNN benchmark, persistent workers, and no startup profiling")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision")
    parser.add_argument("--amp-dtype", type=str, default=None, choices=["float16", "bfloat16"], help="AMP dtype to use when AMP is enabled")
    parser.add_argument("--skip-profile", action="store_true", help="Skip startup FLOPs/inference profiling")
    parser.add_argument("--train-batch-size", type=int, default=None, help="Override the training batch size")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Override the validation batch size")
    parser.add_argument("--test-batch-size", type=int, default=None, help="Override the test batch size")
    parser.add_argument("--prefetch-factor", type=int, default=None, help="Override DataLoader prefetch batches per worker")
    parser.add_argument("--skip-train-metrics", action="store_true", help="Track training loss only; skip expensive per-batch image metrics/correction")
    parser.add_argument("--input-size", type=int, default=None, help="Override square neural-network input size for compatible models such as FC4")
    parser.add_argument("--cache-rgb", action="store_true", help="Cache resized RGB training tensors for fast FC4/RGB training")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory for fast RGB tensor cache")

    opt = parser.parse_args()
    # define the optional args
    args = {}
    if opt.fast:
        args.update({
            "deterministic": False,
            "amp": True,
            "tf32": True,
            "channels_last": True,
            "non_blocking": True,
            "profile_model": False,
            "persistent_workers": True,
            "prefetch_factor": 2,
            "train_metrics": False,
            "cache_rgb": True,
        })
    if opt.device is not None:
        args["device"] = parse_device_arg(opt.device)
    if opt.workers is not None:
        args["n_workers"] = opt.workers
    if opt.amp:
        args["amp"] = True
    if opt.amp_dtype is not None:
        args["amp_dtype"] = opt.amp_dtype
    if opt.skip_profile:
        args["profile_model"] = False
    if opt.train_batch_size is not None:
        args["train_batch_size"] = opt.train_batch_size
    if opt.val_batch_size is not None:
        args["val_batch_size"] = opt.val_batch_size
    if opt.test_batch_size is not None:
        args["test_batch_size"] = opt.test_batch_size
    if opt.prefetch_factor is not None:
        args["prefetch_factor"] = opt.prefetch_factor
    if opt.skip_train_metrics:
        args["train_metrics"] = False
    if opt.input_size is not None:
        args["model_parameter_overrides"] = {"input_size": opt.input_size}
    if opt.cache_rgb:
        args["cache_rgb"] = True
    if opt.cache_dir is not None:
        args["cache_dir"] = opt.cache_dir
    
    # create the runner
    r = Runner(cfg=opt.config_file, **args)
    r.run()