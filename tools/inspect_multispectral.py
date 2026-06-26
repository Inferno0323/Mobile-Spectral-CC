import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import RGBDataset, MSDataset, RGBMSDataset


def load_config(path):
    config = {}
    with open(path, "r") as f:
        exec(f.read(), config)
    return config["cfg"]


def build_dataset(cfg, split):
    is_train = split == "train"
    list_key = f"{split}_list"
    if cfg["data_type"] == "RGB":
        return RGBDataset(
            cfg["dataset_root"],
            cfg[list_key],
            cfg.get("rgb_camera"),
            cfg["gt_type"],
            is_train=is_train,
            seed=cfg["seed"],
        )
    if cfg["data_type"] == "MS":
        return MSDataset(
            cfg["dataset_root"],
            cfg[list_key],
            cfg.get("spectral_camera"),
            cfg["gt_type"],
            is_train=is_train,
            seed=cfg["seed"],
            misaligned=cfg.get("misaligned", False),
        )
    if cfg["data_type"] == "RGB+MS":
        return RGBMSDataset(
            cfg["dataset_root"],
            cfg[list_key],
            cfg.get("rgb_camera"),
            cfg.get("spectral_camera"),
            cfg["gt_type"],
            is_train=is_train,
            seed=cfg["seed"],
            misaligned=cfg.get("misaligned", False),
        )
    raise ValueError(f"Unsupported data_type: {cfg['data_type']}")


def describe_value(name, value):
    if torch.is_tensor(value):
        return f"{name}: tensor shape={tuple(value.shape)} dtype={value.dtype}"
    if isinstance(value, np.ndarray):
        return f"{name}: ndarray shape={value.shape} dtype={value.dtype}"
    if isinstance(value, dict):
        lines = [f"{name}:"]
        for key, item in value.items():
            lines.append("  " + describe_value(key, item))
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        return f"{name}: {type(value).__name__} len={len(value)}"
    return f"{name}: {type(value).__name__}={value}"


def to_numpy_chw(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 4:
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"Expected CHW/HWC image, got shape {value.shape}")
    if value.shape[0] <= 64 and value.shape[0] < value.shape[-1]:
        return value
    return value.transpose(2, 0, 1)


def normalize_image(image):
    image = image.astype(np.float32)
    image = image - np.nanmin(image)
    denom = np.nanmax(image) + 1e-8
    return image / denom


def save_rgb(image_chw, path, title):
    image = normalize_image(image_chw[:3]).transpose(1, 2, 0)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_channel_grid(image_chw, path, title, max_channels=16):
    channels = min(image_chw.shape[0], max_channels)
    cols = min(4, channels)
    rows = int(np.ceil(channels / cols))
    plt.figure(figsize=(3 * cols, 3 * rows))
    for idx in range(channels):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(normalize_image(image_chw[idx]), cmap="viridis")
        ax.set_title(f"ch {idx}")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_spectral_curves(ms_chw, path, title):
    h, w = ms_chw.shape[1:]
    points = {
        "center": (h // 2, w // 2),
        "top_left": (h // 4, w // 4),
        "bottom_right": (3 * h // 4, 3 * w // 4),
    }
    plt.figure(figsize=(8, 5))
    channels = np.arange(ms_chw.shape[0])
    for name, (y, x) in points.items():
        plt.plot(channels, ms_chw[:, y, x], marker="o", label=f"{name} ({x},{y})")
    plt.xlabel("Channel")
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inspect dataset batch shapes and visualize RGB/MS samples.")
    parser.add_argument("config_file", type=str)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="artifacts/inspect_multispectral")
    parser.add_argument("--max-channels", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    dataset = build_dataset(cfg, args.split)
    batch_size = args.batch_size or cfg.get(f"{args.split}_batch_size", cfg.get("train_batch_size", 1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    sample = dataset[args.index]

    os.makedirs(args.out_dir, exist_ok=True)

    print("Batch inspection")
    print("================")
    print(f"config: {args.config_file}")
    print(f"split: {args.split}")
    print(f"dataset length: {len(dataset)}")
    print(f"batch size: {batch_size}")
    for key, value in batch.items():
        print(describe_value(key, value))

    print("\nSingle sample")
    print("=============")
    for key, value in sample.items():
        print(describe_value(key, value))

    if "rgb_image" in sample:
        rgb = to_numpy_chw(sample["rgb_image"])
        save_rgb(rgb, os.path.join(args.out_dir, "rgb_input.png"), "RGB input")

    if "gt_image" in sample:
        gt = to_numpy_chw(sample["gt_image"])
        save_rgb(gt, os.path.join(args.out_dir, "gt_image.png"), "Ground truth")

    if "ms_image" in sample:
        ms = to_numpy_chw(sample["ms_image"])
        save_channel_grid(
            ms,
            os.path.join(args.out_dir, "ms_channels.png"),
            "Multispectral channel grid",
            max_channels=args.max_channels,
        )
        save_spectral_curves(
            ms,
            os.path.join(args.out_dir, "ms_spectral_curves.png"),
            "Multispectral per-pixel channel curves",
        )

    print(f"\nSaved visualizations to {args.out_dir}")


if __name__ == "__main__":
    main()
