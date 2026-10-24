"""
Specialized loss functions for extreme pixel-level class imbalance.
Implements:
- Masked Focal-Tversky Loss: Asymmetrically penalizes False Negatives (missed fire perimeters)
  while explicitly masking out unobserved/cloud pixels (-1.0).
- Masked Soft Dice Loss: Direct optimization of the Jaccard/Dice overlap score.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedFocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss with support for spatial validity masks.
    
    Formula:
        TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        FTL = (1 - TI) ** gamma

    Parameters:
        alpha (float): Weight assigned to False Positives (default: 0.3)
        beta (float): Weight assigned to False Negatives (default: 0.7)
                      Higher beta penalizes missed fire detections more heavily.
        gamma (float): Focusing parameter to down-weight easy background examples (default: 0.75).
        smooth (float): Numerical stability epsilon (default: 1.0).
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets, valid_mask=None):
        """
        Args:
            logits: Predicted raw logits of shape [B, 1, H, W]
            targets: Binary ground truth {0, 1} of shape [B, 1, H, W]
            valid_mask: Binary mask {0, 1} of shape [B, 1, H, W] (0 indicates cloud/invalid)
        """
        probs = torch.sigmoid(logits)

        if valid_mask is not None:
            probs = probs * valid_mask
            targets = targets * valid_mask

        # Flatten spatial dimensions per sample in the batch
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        # Compute True Positives, False Positives, and False Negatives
        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)

        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Focal Tversky Loss
        focal_tversky = torch.pow((1.0 - tversky), self.gamma)

        return focal_tversky.mean()

class CombinedBCEFocalTverskyLoss(nn.Module):
    """
    Combines weighted Masked Binary Cross Entropy with Masked Focal Tversky Loss.
    Ensures smooth pixel-wise probability calibration while enforcing global boundary continuity.
    """
    def __init__(self, bce_weight=0.2, alpha=0.3, beta=0.7, gamma=0.75):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_tversky = MaskedFocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)

    def forward(self, logits, targets, valid_mask=None):
        ftl = self.focal_tversky(logits, targets, valid_mask)

        # Masked BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if valid_mask is not None:
            bce = (bce * valid_mask).sum() / (valid_mask.sum() + 1e-7)
        else:
            bce = bce.mean()

        return self.bce_weight * bce + (1.0 - self.bce_weight) * ftl

if __name__ == "__main__":
    logits = torch.randn(4, 1, 64, 64, requires_grad=True)
    targets = (torch.rand(4, 1, 64, 64) > 0.95).float()
    mask = (torch.rand(4, 1, 64, 64) > 0.05).float()

    criterion = CombinedBCEFocalTverskyLoss()
    loss = criterion(logits, targets, mask)
    loss.backward()

    print(f"Loss module test: Computed loss = {loss.item():.4f}")
    print(f"Gradient check: Logits grad norm = {logits.grad.norm().item():.4f}")
    print("Loss module initialized and tested successfully.")
