"""
Geospatial and mathematical utility functions for Wildfire Spread Prediction.
Includes wind vector decomposition, tensor normalization, and metric computation.
Features defensive guards for zero-fire / single-class edge cases.
"""

import json
import os
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

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
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
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

    # True Positives, False Positives, False Negatives, True Negatives
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

    # PR-AUC & ROC-AUC with edge-case protection
    if len(np.unique(y_true_flat)) < 2:
        # Edge case: All pixels are 0 or all are 1
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
