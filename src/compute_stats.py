"""
Calculates dataset-wide normalization parameters (mean and standard deviation)
strictly from training split to prevent data leakage.
Saves results to configs/scaler_params.json.
"""

import os
import glob
import time
import numpy as np
from dataset import WildfireTFRecordDataset, FEATURE_NAMES
from utils import save_json

def compute_dataset_statistics(raw_dir, num_shards_to_sample=2, output_path="configs/scaler_params.json"):
    train_files = sorted(glob.glob(os.path.join(raw_dir, "*train_*.tfrecord")))[:num_shards_to_sample]
    print(f"Streaming from {len(train_files)} training shards: {[os.path.basename(f) for f in train_files]}")

    ds = WildfireTFRecordDataset(train_files)
    
    # Running statistics per channel using Welford's algorithm / pixel accumulators
    channel_sums = {name: 0.0 for name in FEATURE_NAMES}
    channel_sq_sums = {name: 0.0 for name in FEATURE_NAMES}
    channel_mins = {name: float('inf') for name in FEATURE_NAMES}
    channel_maxs = {name: float('-inf') for name in FEATURE_NAMES}
    total_pixels = 0
    total_samples = 0

    start_time = time.time()
    for inputs, _ in ds:
        # inputs is [12, 64, 64]
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

    # Display clean table
    print("\n--- Training Normalization Statistics ---")
    print(f"{'Feature':<16} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 64)
    for name, s in stats.items():
        print(f"{name:<16} {s['mean']:<12.3f} {s['std']:<12.3f} {s['min']:<12.3f} {s['max']:<12.3f}")

if __name__ == "__main__":
    raw_dir = "/Users/himanshupandey/Desktop/wildfire-spread-prediction/data/raw"
    out_file = "/Users/himanshupandey/Desktop/wildfire-spread-prediction/configs/scaler_params.json"
    compute_dataset_statistics(raw_dir, num_shards_to_sample=2, output_path=out_file)
