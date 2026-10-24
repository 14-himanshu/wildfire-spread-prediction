"""
Calculates dataset-wide normalization parameters (mean and standard deviation)
strictly from training split to prevent data leakage.
Saves results to configs/scaler_params.json.
Uses dynamic cross-platform paths with CLI arguments.
"""

import os
import glob
import time
import argparse
from pathlib import Path
import numpy as np
from dataset import WildfireTFRecordDataset, FEATURE_NAMES
from utils import save_json, DATA_DIR, CONFIGS_DIR

def compute_dataset_statistics(raw_dir=None, num_shards_to_sample=2, output_path=None):
    raw_dir = Path(raw_dir or DATA_DIR)
    output_path = Path(output_path or (CONFIGS_DIR / "scaler_params.json"))

    train_files = sorted(glob.glob(str(raw_dir / "*train_*.tfrecord")))[:num_shards_to_sample]
    print(f"Streaming from {len(train_files)} training shards: {[os.path.basename(f) for f in train_files]}")

    ds = WildfireTFRecordDataset(train_files, normalize=False)
    
    channel_sums = {name: 0.0 for name in FEATURE_NAMES}
    channel_sq_sums = {name: 0.0 for name in FEATURE_NAMES}
    channel_mins = {name: float('inf') for name in FEATURE_NAMES}
    channel_maxs = {name: float('-inf') for name in FEATURE_NAMES}
    total_pixels = 0
    total_samples = 0

    start_time = time.time()
    for inputs, _, _ in ds:
        arr = inputs.numpy()
        pixels_per_patch = arr.shape[1] * arr.shape[2]
        total_pixels += pixels_per_patch
        total_samples += 1

        for idx, name in enumerate(FEATURE_NAMES):
            c_data = arr[idx]
            channel_sums[name] += float(np.sum(c_data))
            channel_sq_sums[name] += float(np.sum(c_data ** 2))
            c_min = float(np.min(c_data))
            c_max = float(np.max(c_data))
            if c_min < channel_mins[name]: channel_mins[name] = c_min
            if c_max > channel_maxs[name]: channel_maxs[name] = c_max

        if total_samples % 500 == 0:
            print(f"Processed {total_samples} samples ({total_pixels:,} pixels)...")

    stats = {}
    for name in FEATURE_NAMES:
        mean = channel_sums[name] / total_pixels
        variance = (channel_sq_sums[name] / total_pixels) - (mean ** 2)
        std = float(np.sqrt(max(variance, 1e-7)))
        stats[name] = {
            "mean": float(mean),
            "std": float(std),
            "min": float(channel_mins[name]),
            "max": float(channel_maxs[name])
        }

    save_json(stats, output_path)
    elapsed = time.time() - start_time
    print(f"\nSuccessfully computed statistics for {total_samples} samples ({total_pixels:,} pixels) in {elapsed:.2f}s")
    print(f"Saved scaler parameters to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute training set channel normalization statistics.")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Path to raw TFRecord files")
    parser.add_argument("--num_shards", type=int, default=2, help="Number of training shards to sample")
    parser.add_argument("--output", type=str, default=str(CONFIGS_DIR / "scaler_params.json"), help="Output path for JSON scaler")
    args = parser.parse_args()

    compute_dataset_statistics(raw_dir=args.data_dir, num_shards_to_sample=args.num_shards, output_path=args.output)
