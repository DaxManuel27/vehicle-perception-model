"""
Inference script for CenterPoint detection.
Loads trained model, runs predictions, decodes boxes.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse

from model import CenterPointModel
from preprocess import points_to_pillars, CONFIG
from dataset import WaymoDataset


def find_local_maxima(heatmap, threshold=0.3, kernel_size=3):
    """
    Find local maxima in heatmap (peak detection).
    
    Args:
        heatmap: (H, W) numpy array
        threshold: minimum confidence
        kernel_size: size of max pooling kernel
    
    Returns:
        List of (x, y, score) tuples
    """
    # Max pooling to find local maxima
    heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0)
    
    pad = kernel_size // 2
    max_pooled = F.max_pool2d(heatmap_tensor, kernel_size, stride=1, padding=pad)
    
    # Points where value equals local max and above threshold
    is_peak = (heatmap_tensor == max_pooled) & (heatmap_tensor > threshold)
    is_peak = is_peak.squeeze().numpy()
    
    # Get coordinates
    peaks = []
    ys, xs = np.where(is_peak)
    for x, y in zip(xs, ys):
        score = heatmap[y, x]
        peaks.append((x, y, score))
    
    # Sort by score descending
    peaks.sort(key=lambda p: p[2], reverse=True)
    
    return peaks


def nms_3d(boxes, iou_threshold=0.5):
    """
    Non-maximum suppression for 3D boxes (simplified BEV NMS).
    
    Args:
        boxes: List of [cx, cy, cz, l, w, h, heading, score]
    
    Returns:
        Filtered list of boxes
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = boxes[:, 7]
    
    # Sort by score
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute BEV IoU (simplified - using center distance)
        cx1, cy1 = boxes[i, 0], boxes[i, 1]
        l1, w1 = boxes[i, 3], boxes[i, 4]
        
        remaining = order[1:]
        cx2, cy2 = boxes[remaining, 0], boxes[remaining, 1]
        l2, w2 = boxes[remaining, 3], boxes[remaining, 4]
        
        # Simple overlap check based on distance
        dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
        size_sum = (l1 + l2 + w1 + w2) / 4
        
        # Keep boxes that are far enough apart
        overlap = dist < size_sum * 0.3
        order = remaining[~overlap]
    
    return [boxes[i].tolist() for i in keep]


def decode_predictions(heatmap, box_pred, config=CONFIG, threshold=0.3, max_boxes=100):
    """
    Decode network outputs into 3D bounding boxes.
    
    Args:
        heatmap: (1, H, W) tensor - detection confidence
        box_pred: (8, H, W) tensor - box parameters
        config: grid configuration
        threshold: confidence threshold
        max_boxes: maximum boxes to return
    
    Returns:
        List of boxes: [cx, cy, cz, length, width, height, heading, score]
    """
    # Convert to numpy
    if isinstance(heatmap, torch.Tensor):
        heatmap = torch.sigmoid(heatmap).cpu().numpy()
    if isinstance(box_pred, torch.Tensor):
        box_pred = box_pred.cpu().numpy()
    
    heatmap = heatmap[0]  # (H, W)
    
    # Find peaks
    peaks = find_local_maxima(heatmap, threshold)[:max_boxes]
    
    boxes = []
    for gx, gy, score in peaks:
        # Extract box parameters
        offset_x = box_pred[0, gy, gx]
        offset_y = box_pred[1, gy, gx]
        z = box_pred[2, gy, gx]
        log_l = box_pred[3, gy, gx]
        log_w = box_pred[4, gy, gx]
        log_h = box_pred[5, gy, gx]
        sin_h = box_pred[6, gy, gx]
        cos_h = box_pred[7, gy, gx]
        
        # Decode size (undo log)
        length = np.exp(log_l)
        width = np.exp(log_w)
        height = np.exp(log_h)
        
        # Decode heading
        heading = np.arctan2(sin_h, cos_h)
        
        # Convert grid coords to world coords
        cx = (gx + offset_x + 0.5) * config['voxel_size'] + config['x_min']
        cy = (gy + offset_y + 0.5) * config['voxel_size'] + config['y_min']
        
        boxes.append([cx, cy, z, length, width, height, heading, float(score)])
    
    # Apply NMS
    boxes = nms_3d(boxes, iou_threshold=0.5)
    
    return boxes


class Detector:
    """Wrapper class for running inference."""
    
    def __init__(self, checkpoint_path, device='cpu'):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: 'cpu', 'cuda', or 'mps'
        """
        self.device = device
        self.config = CONFIG
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = CenterPointModel(CONFIG).to(device)
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    def predict(self, points, threshold=0.1):
        """
        Run detection on a point cloud.
        
        Args:
            points: (N, 4) numpy array [x, y, z, intensity]
            threshold: confidence threshold (lowered to 0.1 for undertrained models)
        
        Returns:
            List of boxes: [cx, cy, cz, length, width, height, heading, score]
        """
        # Preprocess
        pillar_features, pillar_coords, _ = points_to_pillars(points, self.config)
        
        # Add batch dimension
        pillar_features = torch.from_numpy(pillar_features).unsqueeze(0).to(self.device)
        pillar_coords = torch.from_numpy(pillar_coords).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            heatmap, box_pred = self.model(pillar_features, pillar_coords)
        
        # Debug: store max heatmap value
        hm_sigmoid = torch.sigmoid(heatmap[0])
        self.last_max_heatmap = hm_sigmoid.max().item()
        
        # Decode
        boxes = decode_predictions(
            heatmap[0], box_pred[0], 
            self.config, threshold
        )
        
        return boxes


def main():
    parser = argparse.ArgumentParser(description='Run inference on LiDAR data')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Detection threshold')
    parser.add_argument('--frame', type=int, default=0,
                        help='Frame index to run on')
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Load detector
    detector = Detector(args.checkpoint, device)
    
    # Load test data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    dataset = WaymoDataset(data_dir, split='validation')
    
    # Get sample
    sample = dataset[args.frame]
    points = sample['points'].numpy()
    gt_boxes = sample['boxes'].numpy()
    
    print(f"\nFrame {args.frame}:")
    print(f"  Points: {len(points)}")
    print(f"  Ground truth boxes: {len(gt_boxes)}")
    
    # Run inference
    pred_boxes = detector.predict(points, threshold=args.threshold)
    print(f"  Predicted boxes: {len(pred_boxes)}")
    
    # Print predictions
    print("\nPredictions:")
    for i, box in enumerate(pred_boxes):
        cx, cy, cz, l, w, h, heading, score = box
        print(f"  Box {i+1}: center=({cx:.1f}, {cy:.1f}, {cz:.1f}), "
              f"size=({l:.1f}, {w:.1f}, {h:.1f}), score={score:.2f}")


if __name__ == '__main__':
    main()

