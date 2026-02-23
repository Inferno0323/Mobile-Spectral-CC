import argparse
from core.runner import Runner
import ipdb



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="Specify the config file to be used")

    # Optional arguments to override config parameters
    parser.add_argument("--device", type=int, default=None, help="Specify the GPU id to be used (-1 for CPU)")
    parser.add_argument("--workers", type=int, default=None, help="Specify the number of workers")

    opt = parser.parse_args()
    # define the optional args
    args = {}
    if opt.device is not None:
        args["device"] = opt.device
    if opt.workers is not None:
        args["n_workers"] = opt.workers
    
    # create the runner
    r = Runner(cfg=opt.config_file, **args)
    r.run()