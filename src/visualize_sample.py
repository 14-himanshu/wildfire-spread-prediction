"""
Generates high-resolution visualization figures of raw multi-band wildfire tiles
and fire spread progression for documentation and portfolio reports.
Uses portable dynamic paths and CLI arguments.
"""

import os
import glob
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dataset import WildfireTFRecordDataset, FEATURE_NAMES
from utils import decompose_wind, DATA_DIR, REPORTS_DIR

def generate_sample_visualizations(raw_dir=None, output_dir=None):
    raw_dir = Path(raw_dir or DATA_DIR)
    output_dir = Path(output_dir or (REPORTS_DIR / "figures"))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files = sorted(glob.glob(str(raw_dir / "*train_00.tfrecord")))
    ds = WildfireTFRecordDataset(train_files, normalize=False)

    sample_inputs, sample_target = None, None
    for inputs, target, _ in ds:
        prev_fire = inputs[FEATURE_NAMES.index("PrevFireMask")].sum().item()
        next_fire = target.sum().item()
        if prev_fire > 10 and next_fire > prev_fire:
            sample_inputs = inputs.numpy()
            sample_target = target.numpy()[0]
            print(f"Captured active fire tile! Day T burned pixels: {int(prev_fire)}, Day T+1 burned pixels: {int(next_fire)}")
            break

    if sample_inputs is None:
        print("No active fire tile found matching threshold.")
        return

    # 1. Multi-band 12-channel plot
    colormaps = {
        "elevation": "terrain", "pdsi": "BrBG", "NDVI": "YlGn",
        "pr": "Blues", "sph": "PuBu", "th": "hsv",
        "tmmn": "coolwarm", "tmmx": "plasma", "vs": "viridis",
        "erc": "OrRd", "population": "magma", "PrevFireMask": "hot"
    }

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, name in enumerate(FEATURE_NAMES):
        ax = axes[idx]
        data = sample_inputs[idx]
        cmap = colormaps.get(name, "viridis")
        im = ax.imshow(data, cmap=cmap)
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Multimodal Geospatial Bands for Wildfire Tile (64x64 km)", fontsize=16, y=0.98)
    plt.tight_layout()
    out_multiband = output_dir / "multiband_channels.png"
    plt.savefig(out_multiband, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_multiband}")

    # 2. Fire Progression & Wind Vector Overlay Plot
    prev_fire = sample_inputs[FEATURE_NAMES.index("PrevFireMask")]
    elevation = sample_inputs[FEATURE_NAMES.index("elevation")]
    wind_speed = sample_inputs[FEATURE_NAMES.index("vs")]
    wind_dir = sample_inputs[FEATURE_NAMES.index("th")]
    u, v = decompose_wind(wind_speed, wind_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(prev_fire, cmap="hot")
    axes[0].set_title("Day T: Active Perimeter", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(sample_target, cmap="hot")
    axes[1].set_title("Day T+1: Ground Truth Perimeter", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    step = 4
    Y, X = np.mgrid[0:64:step, 0:64:step]
    U_sub = u[::step, ::step]
    V_sub = v[::step, ::step]

    im_elev = axes[2].imshow(elevation, cmap="terrain", alpha=0.85)
    axes[2].contour(prev_fire, levels=[0.5], colors="gold", linewidths=2.0, linestyles="dashed")
    axes[2].contour(sample_target, levels=[0.5], colors="red", linewidths=2.5)
    axes[2].quiver(X, Y, U_sub, V_sub, color="black", scale=40, width=0.005, alpha=0.75)
    axes[2].set_title("Spread vs Wind Vectors & Elevation\n(Gold: Day T, Red: Day T+1)", fontsize=13, fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im_elev, ax=axes[2], fraction=0.046, pad=0.04, label="Elevation (m)")

    plt.tight_layout()
    out_progression = output_dir / "fire_progression_physics.png"
    plt.savefig(out_progression, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_progression}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication figures for wildfire tiles.")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Path to raw TFRecords")
    parser.add_argument("--output_dir", type=str, default=str(REPORTS_DIR / "figures"), help="Output directory for plots")
    args = parser.parse_args()

    generate_sample_visualizations(raw_dir=args.data_dir, output_dir=args.output_dir)
