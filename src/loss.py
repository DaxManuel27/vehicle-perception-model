"""
Loss functions for CenterPoint-style detection.
- Focal loss for heatmap classification
- Smooth L1 loss for box regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for heatmap-based detection.
    Handles extreme class imbalance (background >> foreground).
    """
    
    def __init__(self, alpha=2.0, beta=4.0):
        """
        Args:
            alpha: Focusing parameter for hard examples
            beta: Weight reduction for easy negatives near positives
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) - predicted heatmap (logits)
            target: (B, 1, H, W) - target heatmap (0-1, Gaussian blobs)
        """
        pred = torch.sigmoid(pred)
        pred = pred.clamp(1e-6, 1 - 1e-6)  # Prevent log(0)
        
        # Positive locations (target > 0.99 means center of object)
        pos_mask = target.ge(0.99).float()
        neg_mask = target.lt(0.99).float()
        
        # Positive loss: -log(pred) weighted by (1-pred)^alpha
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred) * pos_mask
        
        # Negative loss: -log(1-pred) weighted by pred^alpha and (1-target)^beta
        # (1-target)^beta reduces loss near positive centers
        neg_loss = -torch.pow(pred, self.alpha) * torch.pow(1 - target, self.beta) * \
                   torch.log(1 - pred) * neg_mask
        
        num_pos = pos_mask.sum()
        
        if num_pos > 0:
            loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        else:
            loss = neg_loss.sum()
        
        return loss


class BoxRegressionLoss(nn.Module):
    """Smooth L1 loss for box regression, only at positive locations."""
    
    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
    
    def forward(self, pred, target, mask):
        """
        Args:
            pred: (B, 8, H, W) - predicted box params
            target: (B, 8, H, W) - target box params
            mask: (B, H, W) - 1.0 at positive locations
        """
        # Expand mask to match channels
        mask = mask.unsqueeze(1).expand_as(pred)
        
        # Compute loss only at positive locations
        loss = self.smooth_l1(pred, target) * mask
        
        num_pos = mask[:, 0, :, :].sum()
        
        if num_pos > 0:
            return loss.sum() / (num_pos * 8)  # Normalize by num_pos and channels
        else:
            return loss.sum() * 0  # Return 0 but keep gradients


class CenterPointLoss(nn.Module):
    """Combined loss for CenterPoint detection."""
    
    def __init__(self, heatmap_weight=1.0, box_weight=2.0):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.box_weight = box_weight
        
        self.focal_loss = FocalLoss(alpha=2.0, beta=4.0)
        self.box_loss = BoxRegressionLoss()
    
    def forward(self, heatmap_pred, box_pred, heatmap_target, box_target, mask):
        """
        Args:
            heatmap_pred: (B, 1, H, W)
            box_pred: (B, 8, H, W)
            heatmap_target: (B, 1, H, W)
            box_target: (B, 8, H, W)
            mask: (B, H, W)
        
        Returns:
            total_loss, heatmap_loss, box_loss
        """
        heatmap_loss = self.focal_loss(heatmap_pred, heatmap_target)
        box_loss = self.box_loss(box_pred, box_target, mask)
        
        total_loss = self.heatmap_weight * heatmap_loss + self.box_weight * box_loss
        
        return total_loss, heatmap_loss, box_loss

