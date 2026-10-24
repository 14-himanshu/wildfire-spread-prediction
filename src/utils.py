"""
Geospatial and mathematical utility functions for Wildfire Spread Prediction.
Includes wind vector decomposition, tensor normalization, metric computation,
and cross-platform project path resolution (zero hard-coded absolute paths).
"""

import json
import os
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

# Dynamic cross-platform project paths (independent of local username)
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

def load_yaml_config(config_path=None):
    import yaml
    if config_path is None:
        config_path = CONFIGS_DIR / "train_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def decompose_wind(wind_speed, wind_direction_deg):
    """
    Decomposes wind speed and direction into zonal (u) and meridional (v) components.
    In meteorology, wind direction indicates WHERE the wind is coming from.
    u: West-to-East component (positive = blowing eastward)
    v: South-to-North component (positive = blowing northward)
    """
    rad = np.deg2rad(wind_direction_deg)
    u = -wind_speed * np.sin(rad)
    v = -wind_speed * np.cos(rad)
    return u, v

def save_json(data, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_binary_metrics(y_true, y_prob, threshold=0.35):
    """
    Computes operational pixel-level classification and segmentation metrics.
    Guarded defensively against single-class / all-unburned tile edge cases.
    """
    y_true_flat = np.nan_to_num(y_true.flatten()).astype(np.uint8)
    y_prob_flat = np.nan_to_num(y_prob.flatten()).astype(np.float32)
    y_pred_flat = (y_prob_flat >= threshold).astype(np.uint8)

    tp = np.sum((y_pred_flat == 1) & (y_true_flat == 1))
    fp = np.sum((y_pred_flat == 1) & (y_true_flat == 0))
    fn = np.sum((y_pred_flat == 0) & (y_true_flat == 1))
    tn = np.sum((y_pred_flat == 0) & (y_true_flat == 0))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    fpr = fp / (fp + tn + 1e-7)

    if len(np.unique(y_true_flat)) < 2:
        pr_auc = 0.0 if np.sum(y_true_flat) == 0 else 1.0
        roc_auc = 0.5
    else:
        try:
            precisions, recalls, _ = precision_recall_curve(y_true_flat, y_prob_flat)
            pr_auc = float(auc(recalls, precisions))
        except Exception:
            pr_auc = 0.0
        
        try:
            roc_auc = float(roc_auc_score(y_true_flat, y_prob_flat))
        except Exception:
            roc_auc = 0.5

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "dice": float(dice),
        "fpr": float(fpr),
        "threshold": float(threshold)
    }
