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

    opt = parser.parse_args()
    # define the optional args
    args = {}
    if opt.device is not None:
        args["device"] = parse_device_arg(opt.device)
    if opt.workers is not None:
        args["n_workers"] = opt.workers
    
    # create the runner
    r = Runner(cfg=opt.config_file, **args)
    r.run()