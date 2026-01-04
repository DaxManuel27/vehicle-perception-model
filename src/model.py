"""
CenterPoint-style 3D object detection model.
Anchor-free detection using heatmaps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocess import CONFIG


class PillarEncoder(nn.Module):
    """Encode pillar features using PointNet-style processing."""
    
    def __init__(self, in_channels=9, out_channels=64):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        """
        Args:
            x: (B, max_pillars, max_points, C)
        Returns:
            (B, max_pillars, out_channels)
        """
        B, P, N, C = x.shape
        
        # Reshape for conv1d: (B*P, C, N)
        x = x.view(B * P, N, C).permute(0, 2, 1)
        
        # Apply conv
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        
        # Max pool over points: (B*P, out_channels)
        x = x.max(dim=2)[0]
        
        # Reshape back: (B, P, out_channels)
        x = x.view(B, P, -1)
        
        return x


class ScatterBEV(nn.Module):
    """Scatter pillar features to BEV grid."""
    
    def __init__(self, config=CONFIG):
        super().__init__()
        self.H = config['grid_h']
        self.W = config['grid_w']
    
    def forward(self, pillar_features, pillar_coords):
        """
        Args:
            pillar_features: (B, max_pillars, C)
            pillar_coords: (B, max_pillars, 2) - [gx, gy]
        Returns:
            (B, C, H, W) BEV feature map
        """
        B, P, C = pillar_features.shape
        device = pillar_features.device
        
        bev = torch.zeros((B, C, self.H, self.W), dtype=pillar_features.dtype, device=device)
        
        for b in range(B):
            coords = pillar_coords[b]  # (P, 2)
            features = pillar_features[b]  # (P, C)
            
            # Valid pillars (non-zero coords or features)
            valid = (coords[:, 0] >= 0) & (coords[:, 0] < self.W) & \
                    (coords[:, 1] >= 0) & (coords[:, 1] < self.H)
            
            gx = coords[valid, 0].long()
            gy = coords[valid, 1].long()
            feat = features[valid]  # (N_valid, C)
            
            # Scatter to BEV
            bev[b, :, gy, gx] = feat.T
        
        return bev


class Backbone(nn.Module):
    """2D CNN backbone for BEV feature extraction."""
    
    def __init__(self, in_channels=64):
        super().__init__()
        
        # Block 1: downsample 2x
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Block 2: downsample 2x
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Block 3: downsample 2x
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # Upsample blocks
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, stride=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 8, stride=8),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
    
    def forward(self, x):
        # Encoder
        x1 = self.block1(x)   # /2
        x2 = self.block2(x1)  # /4
        x3 = self.block3(x2)  # /8
        
        # Decoder - upsample and concat
        u1 = self.up1(x1)
        u2 = self.up2(x2)
        u3 = self.up3(x3)
        
        # Match sizes
        target_size = u1.shape[2:]
        if u2.shape[2:] != target_size:
            u2 = F.interpolate(u2, size=target_size, mode='bilinear', align_corners=False)
        if u3.shape[2:] != target_size:
            u3 = F.interpolate(u3, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate
        out = torch.cat([u1, u2, u3], dim=1)  # 384 channels
        
        return out


class DetectionHead(nn.Module):
    """Detection head: predicts heatmap and box parameters."""
    
    def __init__(self, in_channels=384, num_classes=1):
        super().__init__()
        
        # Shared conv
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        # Heatmap head (classification)
        self.heatmap = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1),
        )
        
        # Box regression head: 8 outputs
        # [offset_x, offset_y, z, log_l, log_w, log_h, sin, cos]
        self.box = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 8, 1),
        )
        
        # Initialize heatmap bias for better training start
        self.heatmap[-1].bias.data.fill_(-2.19)  # -log((1-0.1)/0.1)
    
    def forward(self, x):
        x = self.shared(x)
        heatmap = self.heatmap(x)
        box = self.box(x)
        return heatmap, box


class CenterPointModel(nn.Module):
    """Full CenterPoint-style detection model."""
    
    def __init__(self, config=CONFIG):
        super().__init__()
        self.config = config
        
        self.pillar_encoder = PillarEncoder(in_channels=9, out_channels=64)
        self.scatter = ScatterBEV(config)
        self.backbone = Backbone(in_channels=64)
        self.head = DetectionHead(in_channels=384, num_classes=1)
    
    def forward(self, pillar_features, pillar_coords):
        """
        Args:
            pillar_features: (B, max_pillars, max_points, 9)
            pillar_coords: (B, max_pillars, 2)
        Returns:
            heatmap: (B, 1, H, W) - detection confidence
            box_pred: (B, 8, H, W) - box parameters
        """
        # Encode pillars
        x = self.pillar_encoder(pillar_features)  # (B, P, 64)
        
        # Scatter to BEV
        bev = self.scatter(x, pillar_coords)  # (B, 64, H, W)
        
        # Backbone
        features = self.backbone(bev)  # (B, 384, H/2, W/2)
        
        # Upsample to full resolution
        features = F.interpolate(features, size=(self.config['grid_h'], self.config['grid_w']),
                                  mode='bilinear', align_corners=False)
        
        # Detection head
        heatmap, box_pred = self.head(features)
        
        return heatmap, box_pred


if __name__ == '__main__':
    # Test model
    model = CenterPointModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Dummy input
    B = 2
    pillar_features = torch.randn(B, CONFIG['max_pillars'], CONFIG['max_points_per_pillar'], 9)
    pillar_coords = torch.randint(0, 100, (B, CONFIG['max_pillars'], 2))
    
    heatmap, box_pred = model(pillar_features, pillar_coords)
    print(f"Heatmap shape: {heatmap.shape}")
    print(f"Box pred shape: {box_pred.shape}")

