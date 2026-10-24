"""
Trains and evaluates the Tabular AML Baseline (LightGBM) with spatial neighborhood features.
Features:
- Completely portable paths (zero hardcoded usernames)
- Dynamic CLI arguments and configuration overrides
- Evaluation on untouched held-out validation shards
- Feature importance visualization and sample prediction mapping
"""

import os
import glob
import time
import json
import argparse
from pathlib import Path
import joblib
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from dataset import WildfireTFRecordDataset
from features import extract_tabular_features_from_tile, TABULAR_FEATURE_NAMES
from utils import compute_binary_metrics, save_json, DATA_DIR, MODELS_DIR, REPORTS_DIR, load_yaml_config

def build_tabular_dataset_from_shards(file_paths, max_tiles=400, neg_sample_ratio=6):
    """
    Builds tabular (X, y) matrix from streaming tiles.
    Retains ALL positive burn pixels and subsamples negative unburned pixels
    to balance gradient updates and keep memory low.
    """
    ds = WildfireTFRecordDataset(file_paths, normalize=False)
    
    X_list = []
    y_list = []
    tiles_processed = 0

    print(f"Extracting spatial features from up to {max_tiles} tiles...")
    start_time = time.time()
    for inputs, target, mask in ds:
        inputs_np = inputs.numpy()        # [12, 64, 64]
        target_np = target.numpy()[0]     # [64, 64]
        mask_np = mask.numpy()[0]         # [64, 64]

        # Extract 21 spatial features per pixel: [4096, 21]
        tile_feats = extract_tabular_features_from_tile(inputs_np)
        tile_targets = target_np.flatten().astype(np.uint8)
        tile_valid = mask_np.flatten().astype(bool)

        valid_feats = tile_feats[tile_valid]
        valid_targets = tile_targets[tile_valid]

        pos_idx = np.where(valid_targets == 1)[0]
        neg_idx = np.where(valid_targets == 0)[0]

        if len(pos_idx) > 0:
            n_neg = min(len(neg_idx), len(pos_idx) * neg_sample_ratio)
            sampled_neg_idx = np.random.choice(neg_idx, size=n_neg, replace=False)
            selected_idx = np.concatenate([pos_idx, sampled_neg_idx])
        else:
            if len(neg_idx) > 0:
                selected_idx = np.random.choice(neg_idx, size=min(len(neg_idx), 200), replace=False)
            else:
                continue

        X_list.append(valid_feats[selected_idx])
        y_list.append(valid_targets[selected_idx])

        tiles_processed += 1
        if tiles_processed >= max_tiles:
            break

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    elapsed = time.time() - start_time
    print(f"Extracted {X.shape[0]:,} samples across {tiles_processed} tiles in {elapsed:.2f}s (Positives: {np.sum(y==1):,}, Negatives: {np.sum(y==0):,})")
    return X, y

def evaluate_on_heldout_split(model, eval_files, max_eval_tiles=100):
    ds = WildfireTFRecordDataset(eval_files, normalize=False)
    y_true_all = []
    y_prob_all = []
    latencies = []

    print(f"\nEvaluating baseline on {max_eval_tiles} held-out validation tiles...")
    sample_vis_data = None
    tile_count = 0

    for inputs, target, mask in ds:
        inputs_np = inputs.numpy()
        target_np = target.numpy()[0]
        mask_np = mask.numpy()[0]

        t0 = time.time()
        tile_feats = extract_tabular_features_from_tile(inputs_np)
        probs_flat = model.predict_proba(tile_feats)[:, 1]
        t1 = time.time()
        latencies.append((t1 - t0) * 1000)

        probs_grid = probs_flat.reshape(64, 64)
        valid_mask = mask_np.astype(bool)

        y_true_all.append(target_np[valid_mask])
        y_prob_all.append(probs_grid[valid_mask])

        if sample_vis_data is None and np.sum(target_np) > 25:
            sample_vis_data = (inputs_np, target_np, probs_grid)

        tile_count += 1
        if tile_count >= max_eval_tiles:
            break

    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    metrics = compute_binary_metrics(y_true, y_prob, threshold=best_threshold)
    metrics["avg_tile_inference_ms"] = float(np.mean(latencies))
    metrics["evaluated_tiles"] = tile_count
    metrics["evaluated_pixels"] = len(y_true)
    metrics["positive_fire_pixels"] = int(np.sum(y_true == 1))

    return metrics, sample_vis_data

def train_and_evaluate(data_dir=None, max_train_tiles=600, max_eval_tiles=120, neg_sample_ratio=6):
    data_dir = Path(data_dir or DATA_DIR)
    train_files = sorted(glob.glob(str(data_dir / "*train_*.tfrecord")))[:3]
    eval_files = sorted(glob.glob(str(data_dir / "*eval_*.tfrecord")))

    X_train, y_train = build_tabular_dataset_from_shards(train_files, max_tiles=max_train_tiles, neg_sample_ratio=neg_sample_ratio)

    print("\nTraining LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.2f}s")

    model_path = MODELS_DIR / "lightgbm_baseline.joblib"
    joblib.dump(model, model_path)
    print(f"Saved trained baseline model to: {model_path}")

    metrics, sample_vis = evaluate_on_heldout_split(model, eval_files, max_eval_tiles=max_eval_tiles)
    metrics["train_time_seconds"] = float(train_time)

    metrics_path = REPORTS_DIR / "metrics_tabular_baseline.json"
    save_json(metrics, metrics_path)
    print(f"\n--- Held-Out Benchmark Results ---")
    print(f"  PR-AUC (Precision-Recall AUC): {metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC:                       {metrics['roc_auc']:.4f}")
    print(f"  Optimal Threshold:             {metrics['threshold']:.3f}")
    print(f"  F1-Score at Optimal Threshold: {metrics['f1']:.4f}")
    print(f"  Precision:                     {metrics['precision']:.4f}")
    print(f"  Recall:                        {metrics['recall']:.4f}")
    print(f"  IoU (Jaccard Index):           {metrics['iou']:.4f}")
    print(f"  Dice Score:                    {metrics['dice']:.4f}")
    print(f"  Inference Latency per Tile:    {metrics['avg_tile_inference_ms']:.2f} ms")

    # Feature Importance Figure
    fig, ax = plt.subplots(figsize=(10, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_names = [TABULAR_FEATURE_NAMES[i] for i in indices]

    ax.barh(range(len(indices)), importances[indices][::-1], align="center", color="#e74c3c")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(sorted_names[::-1], fontsize=10)
    ax.set_xlabel("Split Importance (Number of Times Used)", fontweight="bold")
    ax.set_title("LightGBM Baseline: Spatial Feature Importances", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fi_path = REPORTS_DIR / "figures" / "tabular_feature_importance.png"
    plt.savefig(fi_path, dpi=200)
    plt.close()
    print(f"Saved feature importance figure to: {fi_path}")

    # Sample Prediction Visualization
    if sample_vis:
        inputs_np, y_true_tile, y_prob_tile = sample_vis
        prev_fire_tile = inputs_np[11]

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].imshow(prev_fire_tile, cmap="hot")
        axes[0].set_title(f"Day T: Prior Fire (Burned: {int(prev_fire_tile.sum())})", fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(y_true_tile, cmap="hot")
        axes[1].set_title(f"Day T+1: Ground Truth (Burned: {int(y_true_tile.sum())})", fontweight="bold")
        axes[1].axis("off")

        im = axes[2].imshow(y_prob_tile, cmap="inferno", vmin=0, vmax=1)
        axes[2].set_title("LightGBM Predicted Spread Probability", fontweight="bold")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Burn Probability")

        plt.suptitle("LightGBM Baseline: Spatial Wildfire Spread Prediction", fontsize=15, y=0.98)
        plt.tight_layout()
        pred_path = REPORTS_DIR / "figures" / "tabular_sample_prediction.png"
        plt.savefig(pred_path, dpi=200)
        plt.close()
        print(f"Saved sample prediction plot to: {pred_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate LightGBM tabular baseline.")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Path to raw TFRecord files")
    parser.add_argument("--max_train_tiles", type=int, default=600, help="Maximum number of training tiles")
    parser.add_argument("--max_eval_tiles", type=int, default=120, help="Maximum number of evaluation tiles")
    parser.add_argument("--neg_ratio", type=int, default=6, help="Negative to positive pixel sampling ratio")
    args = parser.parse_args()

    train_and_evaluate(
        data_dir=args.data_dir,
        max_train_tiles=args.max_train_tiles,
        max_eval_tiles=args.max_eval_tiles,
        neg_sample_ratio=args.neg_ratio
    )
