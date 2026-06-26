import argparse
import csv
import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np


if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


DATASET_KEY_CANDIDATES = ("spec", "data")
WAVELENGTH_KEY_CANDIDATES = ("wvs", "wavelengths", "wavelength")


def parse_point(value):
    parts = value.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Points must use the format Y,X")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Point coordinates must be integers") from exc


def parse_roi(value):
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROIs must use the format X,Y,W,H")
    try:
        x, y, width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be positive")
    return x, y, width, height


def numeric_dataset_paths(group, prefix=""):
    paths = []
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Dataset):
            if np.issubdtype(item.dtype, np.number) and item.ndim in (2, 3):
                paths.append(path)
        elif isinstance(item, h5py.Group):
            paths.extend(numeric_dataset_paths(item, path))
    return paths


def choose_dataset_key(handle, requested_key=None):
    if requested_key:
        if requested_key not in handle:
            raise KeyError(f"Dataset key '{requested_key}' was not found in {handle.filename}")
        return requested_key

    for key in DATASET_KEY_CANDIDATES:
        if key in handle and isinstance(handle[key], h5py.Dataset):
            return key

    candidates = numeric_dataset_paths(handle)
    three_dimensional = [path for path in candidates if handle[path].ndim == 3]
    if three_dimensional:
        return three_dimensional[0]
    if candidates:
        return candidates[0]

    raise KeyError(f"No numeric 2D or 3D dataset was found in {handle.filename}")


def choose_wavelength_key(handle, requested_key=None):
    if requested_key:
        if requested_key not in handle:
            raise KeyError(f"Wavelength key '{requested_key}' was not found in {handle.filename}")
        return requested_key

    for key in WAVELENGTH_KEY_CANDIDATES:
        if key in handle and isinstance(handle[key], h5py.Dataset):
            return key

    return None


def infer_channel_axis(array, wavelengths, channel_axis):
    if array.ndim == 2:
        return array[None, :, :]
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D or 3D image array, got shape {array.shape}")

    if channel_axis == "first":
        return array
    if channel_axis == "last":
        return np.moveaxis(array, -1, 0)

    if wavelengths is not None:
        if len(wavelengths) == array.shape[0]:
            return array
        if len(wavelengths) == array.shape[-1]:
            return np.moveaxis(array, -1, 0)

    first_axis_looks_like_channels = array.shape[0] <= 64 and array.shape[0] <= min(array.shape[1:])
    last_axis_looks_like_channels = array.shape[-1] <= 64 and array.shape[-1] <= min(array.shape[:2])
    if first_axis_looks_like_channels and not last_axis_looks_like_channels:
        return array
    if last_axis_looks_like_channels and not first_axis_looks_like_channels:
        return np.moveaxis(array, -1, 0)

    return array


def load_spectral_image(path, dataset_key=None, wavelength_key=None, channel_axis="auto"):
    with h5py.File(path, "r") as handle:
        data_key = choose_dataset_key(handle, dataset_key)
        wavelengths_key = choose_wavelength_key(handle, wavelength_key)

        data = np.asarray(handle[data_key], dtype=np.float32)
        wavelengths = None
        if wavelengths_key:
            wavelengths = np.asarray(handle[wavelengths_key], dtype=np.float32).reshape(-1)

    data = infer_channel_axis(data, wavelengths, channel_axis)
    channels = data.shape[0]

    if wavelengths is None or len(wavelengths) != channels:
        wavelengths = np.arange(channels, dtype=np.float32)

    return data, wavelengths, data_key


def channel_label(index, wavelengths):
    wavelength = wavelengths[index]
    if np.isclose(wavelength, round(float(wavelength))):
        wavelength = int(round(float(wavelength)))
    return f"C{index} ({wavelength} nm)"


def normalize_channel(channel, percentiles):
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        return np.zeros_like(channel, dtype=np.float32)

    low, high = np.percentile(finite, percentiles)
    if high <= low:
        return np.zeros_like(channel, dtype=np.float32)

    normalized = (channel - low) / (high - low)
    return np.clip(normalized, 0, 1)


def selected_channel_indices(num_channels, max_channels):
    if max_channels is None or max_channels >= num_channels:
        return np.arange(num_channels)
    return np.unique(np.linspace(0, num_channels - 1, max_channels).round().astype(int))


def validate_point(point, height, width):
    y, x = point
    if y < 0 or y >= height or x < 0 or x >= width:
        raise ValueError(f"Point {y},{x} is outside image bounds H={height}, W={width}")


def validate_roi(roi, height, width):
    x, y, roi_width, roi_height = roi
    if x < 0 or y < 0 or x + roi_width > width or y + roi_height > height:
        raise ValueError(f"ROI {x},{y},{roi_width},{roi_height} is outside image bounds H={height}, W={width}")


def collect_profiles(data, points, rois, normalize_profiles):
    _, height, width = data.shape
    profiles = [("image mean", data.mean(axis=(1, 2)))]

    if not points and not rois:
        points = [(height // 2, width // 2)]

    for point in points:
        validate_point(point, height, width)
        y, x = point
        profiles.append((f"pixel y={y}, x={x}", data[:, y, x]))

    for roi in rois:
        validate_roi(roi, height, width)
        x, y, roi_width, roi_height = roi
        roi_data = data[:, y:y + roi_height, x:x + roi_width]
        profiles.append((f"roi x={x}, y={y}, w={roi_width}, h={roi_height}", roi_data.mean(axis=(1, 2))))

    if normalize_profiles:
        normalized_profiles = []
        for label, values in profiles:
            low = np.nanmin(values)
            high = np.nanmax(values)
            if high > low:
                values = (values - low) / (high - low)
            else:
                values = np.zeros_like(values)
            normalized_profiles.append((label, values))
        profiles = normalized_profiles

    return profiles


def per_channel_stats(data, wavelengths, percentiles):
    stats = []
    for index, channel in enumerate(data):
        finite = channel[np.isfinite(channel)]
        if finite.size:
            low, high = np.percentile(finite, percentiles)
            row = {
                "channel": index,
                "wavelength": float(wavelengths[index]),
                "min": float(np.min(finite)),
                "percentile_low": float(low),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "percentile_high": float(high),
                "max": float(np.max(finite)),
            }
        else:
            row = {
                "channel": index,
                "wavelength": float(wavelengths[index]),
                "min": np.nan,
                "percentile_low": np.nan,
                "mean": np.nan,
                "std": np.nan,
                "percentile_high": np.nan,
                "max": np.nan,
            }
        stats.append(row)
    return stats


def write_stats_csv(path, stats):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)


def make_stats_table(axis, stats, channel_indices):
    axis.axis("off")
    table_rows = []
    for index in channel_indices:
        row = stats[index]
        table_rows.append([
            int(row["channel"]),
            f"{row['wavelength']:.0f}",
            f"{row['min']:.4g}",
            f"{row['mean']:.4g}",
            f"{row['std']:.4g}",
            f"{row['max']:.4g}",
        ])

    table = axis.table(
        cellText=table_rows,
        colLabels=("channel", "wavelength", "min", "mean", "std", "max"),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)


def render_view(
    data,
    wavelengths,
    image_path,
    dataset_key,
    points,
    rois,
    max_channels,
    columns,
    cmap,
    percentiles,
    normalize_profiles,
):
    num_channels, height, width = data.shape
    channel_indices = selected_channel_indices(num_channels, max_channels)
    rows = int(np.ceil(len(channel_indices) / columns))
    stats = per_channel_stats(data, wavelengths, percentiles)
    profiles = collect_profiles(data, points, rois, normalize_profiles)

    figure_height = max(7, rows * 2.1 + 4.8)
    figure_width = max(8, columns * 2.6)
    figure = plt.figure(figsize=(figure_width, figure_height), constrained_layout=True)
    grid = figure.add_gridspec(rows + 2, columns, height_ratios=[1] * rows + [1.15, 0.85])

    for slot, channel_index in enumerate(channel_indices):
        axis = figure.add_subplot(grid[slot // columns, slot % columns])
        axis.imshow(normalize_channel(data[channel_index], percentiles), cmap=cmap)
        axis.set_title(channel_label(channel_index, wavelengths), fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    for slot in range(len(channel_indices), rows * columns):
        axis = figure.add_subplot(grid[slot // columns, slot % columns])
        axis.axis("off")

    profile_axis = figure.add_subplot(grid[rows, :])
    for label, values in profiles:
        profile_axis.plot(wavelengths, values, marker="o", linewidth=1.4, markersize=3, label=label)
    profile_axis.set_xlabel("Wavelength / channel")
    profile_axis.set_ylabel("Normalized value" if normalize_profiles else "Value")
    profile_axis.grid(True, alpha=0.25)
    profile_axis.legend(loc="best", fontsize=8)

    table_axis = figure.add_subplot(grid[rows + 1, :])
    make_stats_table(table_axis, stats, channel_indices)

    figure.suptitle(
        f"{Path(image_path).name}  |  dataset='{dataset_key}'  |  shape=(C={num_channels}, H={height}, W={width})",
        fontsize=12,
    )
    return figure, stats


def build_parser():
    parser = argparse.ArgumentParser(
        description="Visualize spectral HDF5 images as channel grids, spectral profiles, and per-channel stats.",
    )
    parser.add_argument("image", help="Path to a spectral .h5 image")
    parser.add_argument("--key", default=None, help="HDF5 dataset key to read (defaults to spec, data, or first numeric 3D dataset)")
    parser.add_argument("--wavelength-key", default=None, help="HDF5 wavelength key to read (defaults to wvs/wavelengths when present)")
    parser.add_argument(
        "--channel-axis",
        choices=("auto", "first", "last"),
        default="auto",
        help="Where the spectral/channel axis is stored in the HDF5 array",
    )
    parser.add_argument("--output", "-o", default=None, help="Save the visualization to this image path instead of opening a window")
    parser.add_argument("--stats-output", default=None, help="Optional CSV path for per-channel statistics")
    parser.add_argument("--max-channels", type=int, default=16, help="Maximum number of channel thumbnails to draw")
    parser.add_argument("--columns", type=int, default=4, help="Number of columns in the channel thumbnail grid")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap for channel thumbnails")
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs=2,
        default=(1.0, 99.0),
        metavar=("LOW", "HIGH"),
        help="Percentiles used for thumbnail contrast and reported percentile columns",
    )
    parser.add_argument(
        "--point",
        type=parse_point,
        action="append",
        default=[],
        help="Add a spectral profile for a pixel using Y,X. Can be provided more than once.",
    )
    parser.add_argument(
        "--roi",
        type=parse_roi,
        action="append",
        default=[],
        help="Add a mean spectral profile for a region using X,Y,W,H. Can be provided more than once.",
    )
    parser.add_argument(
        "--normalize-profiles",
        action="store_true",
        help="Normalize each plotted profile independently to [0, 1] to compare spectral shapes.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Output figure DPI")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.max_channels <= 0:
        parser.error("--max-channels must be positive")
    if args.columns <= 0:
        parser.error("--columns must be positive")
    if args.percentiles[0] >= args.percentiles[1]:
        parser.error("--percentiles LOW must be smaller than HIGH")

    data, wavelengths, dataset_key = load_spectral_image(
        args.image,
        dataset_key=args.key,
        wavelength_key=args.wavelength_key,
        channel_axis=args.channel_axis,
    )
    figure, stats = render_view(
        data=data,
        wavelengths=wavelengths,
        image_path=args.image,
        dataset_key=dataset_key,
        points=args.point,
        rois=args.roi,
        max_channels=args.max_channels,
        columns=args.columns,
        cmap=args.cmap,
        percentiles=args.percentiles,
        normalize_profiles=args.normalize_profiles,
    )

    if args.stats_output:
        write_stats_csv(args.stats_output, stats)
        print(f"Wrote channel statistics to {args.stats_output}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=args.dpi)
        print(f"Wrote spectral visualization to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
