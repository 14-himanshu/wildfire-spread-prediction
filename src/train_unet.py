"""
PyTorch training and evaluation engine for WildfireUNet.
Features:
- Hardware acceleration (MPS on Apple Silicon, CUDA on NVIDIA, or CPU)
- Streaming TFRecord dataset loader with zero-leakage normalization
- Masked Focal-Tversky loss targeting severe 99.7% class imbalance
- Model checkpointing and quantitative held-out benchmarking
- Visual sample prediction rendering comparing ground truth with spatial predictions
"""

import os
import glob
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import WildfireTFRecordDataset
from models.unet import WildfireUNet
from loss import CombinedBCEFocalTverskyLoss
from utils import compute_binary_metrics, save_json

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def evaluate_unet(model, val_loader, criterion, device, max_batches=25):
    model.eval()
    val_loss = 0.0
    y_true_list = []
    y_prob_list = []
    sample_vis = None
    latencies = []

    with torch.no_grad():
        for b_idx, (inputs, targets, masks) in enumerate(val_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            t0 = time.time()
            logits = model(inputs)
            latencies.append((time.time() - t0) * 1000 / inputs.size(0))

            loss = criterion(logits, targets, masks)
            val_loss += loss.item()

            probs = torch.sigmoid(logits)

            # Filter valid observable pixels
            targets_np = targets.cpu().numpy()
            probs_np = probs.cpu().numpy()
            masks_np = masks.cpu().numpy().astype(bool)

            valid_targets = targets_np[masks_np]
            valid_probs = probs_np[masks_np]

            y_true_list.append(valid_targets)
            y_prob_list.append(valid_probs)

            # Capture visual sample with active burning
            if sample_vis is None:
                for s_i in range(inputs.size(0)):
                    if targets_np[s_i, 0].sum() > 20:
                        sample_vis = (
                            inputs[s_i].cpu().numpy(),
                            targets_np[s_i, 0],
                            probs_np[s_i, 0]
                        )
                        break

            if b_idx + 1 >= max_batches:
                break

    y_true = np.concatenate(y_true_list)
    y_prob = np.concatenate(y_prob_list)

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.35)
    metrics["val_loss"] = val_loss / max(1, b_idx + 1)
    metrics["avg_tile_inference_ms"] = float(np.mean(latencies))
    metrics["evaluated_pixels"] = len(y_true)
    metrics["evaluated_tiles"] = (b_idx + 1) * val_loader.batch_size

    return metrics, sample_vis

def train_unet(epochs=3, batch_size=16, lr=5e-4, num_train_shards=2):
    device = get_device()
    print(f"Initializing U-Net training pipeline on device: {device}")

    data_dir = "/Users/himanshupandey/Desktop/wildfire-spread-prediction/data/raw"
    scaler_file = "/Users/himanshupandey/Desktop/wildfire-spread-prediction/configs/scaler_params.json"
    train_files = sorted(glob.glob(os.path.join(data_dir, "*train_*.tfrecord")))[:num_train_shards]
    eval_files = sorted(glob.glob(os.path.join(data_dir, "*eval_*.tfrecord")))

    print(f"Training shards: {[os.path.basename(f) for f in train_files]}")
    print(f"Validation shards: {[os.path.basename(f) for f in eval_files]}")

    model = WildfireUNet(in_channels=12, out_channels=1, base_channels=32).to(device)
    criterion = CombinedBCEFocalTverskyLoss(bce_weight=0.2, alpha=0.3, beta=0.7, gamma=0.75)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_pr_auc = -1.0
    best_metrics = {}
    best_vis = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_ds = WildfireTFRecordDataset(train_files, scaler_path=scaler_file, normalize=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size)

        epoch_loss = 0.0
        batch_count = 0
        t0 = time.time()

        for inputs, targets, masks in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            if batch_count % 30 == 0:
                print(f"  Epoch [{epoch}/{epochs}] Batch {batch_count}: Loss = {loss.item():.4f}")

            # Cap local training batches per epoch for prototyping
            if batch_count >= 120:
                break

        scheduler.step()
        train_time = time.time() - t0
        avg_train_loss = epoch_loss / max(1, batch_count)

        # Validation pass
        val_ds = WildfireTFRecordDataset(eval_files, scaler_path=scaler_file, normalize=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        val_metrics, sample_vis = evaluate_unet(model, val_loader, criterion, device, max_batches=20)

        print(f"\nEpoch {epoch}/{epochs} Summary (Time: {train_time:.1f}s):")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"  Val PR-AUC: {val_metrics['pr_auc']:.4f} | Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        print(f"  Val F1:     {val_metrics['f1']:.4f}     | Val Dice:    {val_metrics['dice']:.4f}")
        print(f"  Val IoU:    {val_metrics['iou']:.4f}     | Latency:     {val_metrics['avg_tile_inference_ms']:.2f} ms/tile\n")

        if val_metrics['pr_auc'] > best_pr_auc or epoch == epochs:
            best_pr_auc = val_metrics['pr_auc']
            best_metrics = val_metrics
            best_vis = sample_vis
            
            # Save checkpoint
            ckpt_path = "/Users/himanshupandey/Desktop/wildfire-spread-prediction/models/wildfire_unet_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics
            }, ckpt_path)
            print(f"  [Checkpoint] Saved best U-Net model to: {ckpt_path}")

    # Save metrics JSON
    metrics_path = "/Users/himanshupandey/Desktop/wildfire-spread-prediction/reports/metrics_unet.json"
    save_json(best_metrics, metrics_path)
    print(f"Saved evaluation metrics to: {metrics_path}")

    # Generate sample prediction figure
    if best_vis:
        inputs_np, y_true_tile, y_prob_tile = best_vis
        prev_fire_tile = inputs_np[11] # PrevFireMask

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].imshow(prev_fire_tile, cmap="hot")
        axes[0].set_title(f"Day T: Prior Fire Perimeter", fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(y_true_tile, cmap="hot")
        axes[1].set_title(f"Day T+1: Ground Truth Fire Front", fontweight="bold")
        axes[1].axis("off")

        im = axes[2].imshow(y_prob_tile, cmap="inferno", vmin=0, vmax=1)
        axes[2].set_title("Spatial U-Net Predicted Probability Map", fontweight="bold")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Burn Probability")

        plt.suptitle("PyTorch Spatial U-Net: Next Day Wildfire Spread Prediction", fontsize=15, y=0.98)
        plt.tight_layout()
        pred_path = "/Users/himanshupandey/Desktop/wildfire-spread-prediction/reports/figures/unet_sample_prediction.png"
        plt.savefig(pred_path, dpi=200)
        plt.close()
        print(f"Saved U-Net prediction figure to: {pred_path}")

    return best_metrics

if __name__ == "__main__":
    train_unet(epochs=3, batch_size=16, lr=5e-4, num_train_shards=2)
