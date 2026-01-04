"""
Preprocessing: Convert point clouds to pillar features for PointPillars.
Also creates target tensors from ground truth boxes.
"""

import torch
import numpy as np
from collections import defaultdict


# Grid configuration
CONFIG = {
    'x_min': -50.0,
    'x_max': 50.0,
    'y_min': -50.0,
    'y_max': 50.0,
    'z_min': -3.0,
    'z_max': 3.0,
    'voxel_size': 0.25,  # meters per cell
    'max_pillars': 12000,
    'max_points_per_pillar': 32,
}

# Derived values
CONFIG['grid_w'] = int((CONFIG['x_max'] - CONFIG['x_min']) / CONFIG['voxel_size'])  # 400
CONFIG['grid_h'] = int((CONFIG['y_max'] - CONFIG['y_min']) / CONFIG['voxel_size'])  # 400


def points_to_pillars(points, config=CONFIG):
    """
    Convert point cloud to pillar representation.
    
    Args:
        points: (N, 4) tensor [x, y, z, intensity]
        config: grid configuration
    
    Returns:
        pillar_features: (max_pillars, max_points, 9) - point features per pillar
        pillar_coords: (max_pillars, 2) - grid coordinates [gx, gy]
        num_pillars: actual number of non-empty pillars
    """
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    
    # Filter points within range
    mask = (
        (points[:, 0] >= config['x_min']) & (points[:, 0] < config['x_max']) &
        (points[:, 1] >= config['y_min']) & (points[:, 1] < config['y_max']) &
        (points[:, 2] >= config['z_min']) & (points[:, 2] < config['z_max'])
    )
    points = points[mask]
    
    if len(points) == 0:
        return (
            np.zeros((config['max_pillars'], config['max_points_per_pillar'], 9), dtype=np.float32),
            np.zeros((config['max_pillars'], 2), dtype=np.int32),
            0
        )
    
    # Compute grid indices
    gx = ((points[:, 0] - config['x_min']) / config['voxel_size']).astype(np.int32)
    gy = ((points[:, 1] - config['y_min']) / config['voxel_size']).astype(np.int32)
    
    # Group points by pillar
    pillars = defaultdict(list)
    for i, (x, y) in enumerate(zip(gx, gy)):
        pillars[(x, y)].append(i)
    
    # Create output tensors
    max_pillars = config['max_pillars']
    max_points = config['max_points_per_pillar']
    
    pillar_features = np.zeros((max_pillars, max_points, 9), dtype=np.float32)
    pillar_coords = np.zeros((max_pillars, 2), dtype=np.int32)
    
    pillar_idx = 0
    for (px, py), point_indices in pillars.items():
        if pillar_idx >= max_pillars:
            break
        
        # Get points in this pillar
        pts = points[point_indices[:max_points]]  # Limit points
        n_pts = len(pts)
        
        # Compute pillar center
        mean_xyz = pts[:, :3].mean(axis=0)
        
        # Pillar cell center in world coords
        cell_center_x = config['x_min'] + (px + 0.5) * config['voxel_size']
        cell_center_y = config['y_min'] + (py + 0.5) * config['voxel_size']
        
        # Build 9-dim features per point:
        # [x, y, z, intensity, x-mean, y-mean, z-mean, x-cell, y-cell]
        for i, (x, y, z, intensity) in enumerate(pts):
            pillar_features[pillar_idx, i] = [
                x, y, z, intensity,
                x - mean_xyz[0], y - mean_xyz[1], z - mean_xyz[2],
                x - cell_center_x, y - cell_center_y
            ]
        
        pillar_coords[pillar_idx] = [px, py]
        pillar_idx += 1
    
    return pillar_features, pillar_coords, pillar_idx


def create_targets(boxes, config=CONFIG):
    """
    Create training targets from ground truth boxes (CenterPoint style).
    
    Uses Gaussian heatmaps for classification instead of hard labels.
    
    Args:
        boxes: (N, 7) array [cx, cy, cz, length, width, height, heading]
        config: grid configuration
    
    Returns:
        heatmap: (1, H, W) - Gaussian heatmap for car centers
        box_targets: (7, H, W) - regression targets [cx, cy, cz, log(l), log(w), log(h), sin, cos]
        mask: (H, W) - 1.0 at positive locations
    """
    H = config['grid_h']
    W = config['grid_w']
    
    heatmap = np.zeros((1, H, W), dtype=np.float32)
    box_targets = np.zeros((8, H, W), dtype=np.float32)  # 8: cx,cy,cz,log_l,log_w,log_h,sin,cos
    mask = np.zeros((H, W), dtype=np.float32)
    
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()
    
    for box in boxes:
        cx, cy, cz, length, width, height, heading = box
        
        # Convert to grid coords
        gx = (cx - config['x_min']) / config['voxel_size']
        gy = (cy - config['y_min']) / config['voxel_size']
        
        gx_int = int(gx)
        gy_int = int(gy)
        
        if not (0 <= gx_int < W and 0 <= gy_int < H):
            continue
        
        # Create Gaussian heatmap (radius based on object size)
        radius = max(1, int(min(length, width) / config['voxel_size'] / 2))
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = gx_int + dx, gy_int + dy
                if 0 <= nx < W and 0 <= ny < H:
                    dist_sq = dx*dx + dy*dy
                    sigma = radius / 3.0
                    value = np.exp(-dist_sq / (2 * sigma * sigma))
                    heatmap[0, ny, nx] = max(heatmap[0, ny, nx], value)
        
        # Regression targets at center cell
        # Offset from cell center
        offset_x = gx - gx_int - 0.5
        offset_y = gy - gy_int - 0.5
        
        box_targets[0, gy_int, gx_int] = offset_x  # sub-cell x offset
        box_targets[1, gy_int, gx_int] = offset_y  # sub-cell y offset
        box_targets[2, gy_int, gx_int] = cz        # z center
        box_targets[3, gy_int, gx_int] = np.log(length + 1e-6)  # log size
        box_targets[4, gy_int, gx_int] = np.log(width + 1e-6)
        box_targets[5, gy_int, gx_int] = np.log(height + 1e-6)
        box_targets[6, gy_int, gx_int] = np.sin(heading)  # sin/cos encoding
        box_targets[7, gy_int, gx_int] = np.cos(heading)
        
        mask[gy_int, gx_int] = 1.0
    
    return heatmap, box_targets, mask


def batch_preprocess(batch, config=CONFIG, device='cpu'):
    """
    Preprocess a batch of data for training.
    
    Returns:
        pillar_features: (B, max_pillars, max_points, 9)
        pillar_coords: (B, max_pillars, 2)
        heatmaps: (B, 1, H, W)
        box_targets: (B, 8, H, W)
        masks: (B, H, W)
    """
    batch_size = len(batch['points'])
    
    all_features = []
    all_coords = []
    all_heatmaps = []
    all_box_targets = []
    all_masks = []
    
    for i in range(batch_size):
        points = batch['points'][i]
        boxes = batch['boxes'][i]
        
        # Points to pillars
        features, coords, _ = points_to_pillars(points, config)
        all_features.append(features)
        all_coords.append(coords)
        
        # Create targets
        heatmap, box_target, mask = create_targets(boxes, config)
        all_heatmaps.append(heatmap)
        all_box_targets.append(box_target)
        all_masks.append(mask)
    
    return {
        'pillar_features': torch.from_numpy(np.stack(all_features)).to(device),
        'pillar_coords': torch.from_numpy(np.stack(all_coords)).to(device),
        'heatmaps': torch.from_numpy(np.stack(all_heatmaps)).to(device),
        'box_targets': torch.from_numpy(np.stack(all_box_targets)).to(device),
        'masks': torch.from_numpy(np.stack(all_masks)).to(device),
    }

