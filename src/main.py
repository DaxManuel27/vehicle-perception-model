#!/usr/bin/env python3
"""
Waymo LiDAR Visualizer
Navigate through frames with arrow keys.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def range_image_to_point_cloud(range_image, laser_name):
    """Convert range image to 3D point cloud"""
    height, width = range_image.shape[0], range_image.shape[1]
    range_values = range_image[:, :, 0]
    valid_mask = range_values > 0
    
    if np.sum(valid_mask) == 0:
        return np.array([]).reshape(0, 3)
    
    if laser_name == 0:  # TOP
        inclination = np.linspace(-np.pi/6, np.pi/6, height)
    else:
        inclination = np.linspace(-np.pi/8, np.pi/8, height)
    
    azimuth = np.linspace(-np.pi, np.pi, width)
    azimuth_grid, inclination_grid = np.meshgrid(azimuth, inclination)
    
    x = range_values * np.cos(inclination_grid) * np.cos(azimuth_grid)
    y = range_values * np.cos(inclination_grid) * np.sin(azimuth_grid)
    z = range_values * np.sin(inclination_grid)
    
    points = np.stack([x, y, z], axis=-1)
    return points[valid_mask]


def load_frame(lidar_df, box_df, timestamp):
    """Load point cloud and boxes for a single frame"""
    frame_data = lidar_df[lidar_df['key.frame_timestamp_micros'] == timestamp]
    
    all_points = []
    for idx, row in frame_data.iterrows():
        range_image_bytes = row['[LiDARComponent].range_image_return1.values']
        range_image_shape = row['[LiDARComponent].range_image_return1.shape']
        
        range_image = np.frombuffer(range_image_bytes, dtype=np.float32)
        shape = tuple(range_image_shape) if isinstance(range_image_shape, (list, tuple)) else range_image_shape
        range_image = range_image.reshape(shape)
        
        laser_name = row['key.laser_name']
        points = range_image_to_point_cloud(range_image, laser_name)
        if len(points) > 0:
            all_points.append(points)
    
    combined_points = np.vstack(all_points) if all_points else np.zeros((0, 3))
    
    # Get boxes
    boxes = []
    box_frame = box_df[box_df['key.frame_timestamp_micros'] == timestamp]
    box_center_col = '[LiDARBoxComponent].box.center.x'
    
    if box_center_col in box_frame.columns:
        box_rows = box_frame[box_frame[box_center_col].notna()]
        for idx, row in box_rows.iterrows():
            if row['[LiDARBoxComponent].type'] == 1:  # Vehicles only
                boxes.append({
                    'center': np.array([row['[LiDARBoxComponent].box.center.x'],
                                       -row['[LiDARBoxComponent].box.center.y'],
                                        row['[LiDARBoxComponent].box.center.z']]),
                    'size': np.array([row['[LiDARBoxComponent].box.size.x'],
                                     row['[LiDARBoxComponent].box.size.y'],
                                     row['[LiDARBoxComponent].box.size.z']]),
                    'heading': -row['[LiDARBoxComponent].box.heading']
                })
    
    return combined_points, boxes


def draw_box_3d(ax, box):
    """Draw a 3D bounding box"""
    center = box['center']
    size = box['size']
    heading = box['heading']
    
    dx, dy, dz = size[0]/2, size[1]/2, size[2]/2
    
    corners = np.array([
        [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
        [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz]
    ])
    
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    rot = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])
    corners = corners @ rot.T + center
    
    # Draw edges
    edges = [
        [0,1], [1,2], [2,3], [3,0],  # bottom
        [4,5], [5,6], [6,7], [7,4],  # top
        [0,4], [1,5], [2,6], [3,7]   # vertical
    ]
    for e in edges:
        ax.plot3D(*zip(corners[e[0]], corners[e[1]]), 'r-', linewidth=1)


class LiDARViewer:
    def __init__(self, lidar_df, box_df, subsample=10, show_boxes=True, detector=None):
        self.lidar_df = lidar_df
        self.box_df = box_df
        self.subsample = subsample
        self.show_boxes = show_boxes
        self.detector = detector  # Optional: for showing predictions
        self.timestamps = sorted(lidar_df['key.frame_timestamp_micros'].unique())
        self.total_frames = len(self.timestamps)
        self.current_frame = 0
        
        # Create figure
        self.fig = plt.figure(figsize=(14, 9), facecolor='black')
        self.ax = self.fig.add_subplot(111, projection='3d', facecolor='black')
        
        # Set initial view
        self.ax.set_xlim([-50, 50])
        self.ax.set_ylim([-50, 50])
        self.ax.set_zlim([-5, 10])
        self.ax.set_axis_off()
        self.ax.grid(False)
        self.ax.view_init(elev=30, azim=-60)
        
        # Store artists for removal
        self.scatter = None
        self.box_lines = []
        self.pred_lines = []
        
        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Draw first frame
        self.update_frame()
        
    def on_scroll(self, event):
        """Handle mouse scroll for zoom"""
        if event.button == 'up':
            self.ax._dist = max(3, self.ax._dist * 0.9)
        elif event.button == 'down':
            self.ax._dist = min(30, self.ax._dist * 1.1)
        self.fig.canvas.draw_idle()
        
    def on_key(self, event):
        """Handle keyboard input"""
        if event.key == 'right':
            self.current_frame = (self.current_frame + 1) % self.total_frames
            self.update_frame()
        elif event.key == 'left':
            self.current_frame = (self.current_frame - 1) % self.total_frames
            self.update_frame()
        elif event.key == 'up':
            self.current_frame = min(self.current_frame + 10, self.total_frames - 1)
            self.update_frame()
        elif event.key == 'down':
            self.current_frame = max(self.current_frame - 10, 0)
            self.update_frame()
        elif event.key in ['=', '+']:
            self.ax._dist = max(3, self.ax._dist * 0.8)
            self.fig.canvas.draw_idle()
        elif event.key in ['-', '_']:
            self.ax._dist = min(30, self.ax._dist * 1.2)
            self.fig.canvas.draw_idle()
        elif event.key == 'r':
            self.ax._dist = 10
            self.ax.view_init(elev=30, azim=-60)
            self.fig.canvas.draw_idle()
        elif event.key == 'q':
            plt.close(self.fig)
    
    def update_frame(self):
        """Update the visualization with current frame - preserves view"""
        # Remove old scatter
        if self.scatter is not None:
            self.scatter.remove()
        
        # Remove old box lines
        for line in self.box_lines:
            line.remove()
        self.box_lines = []
        
        # Remove old prediction lines
        for line in self.pred_lines:
            line.remove()
        self.pred_lines = []
        
        # Load frame data
        timestamp = self.timestamps[self.current_frame]
        points, boxes = load_frame(self.lidar_df, self.box_df, timestamp)
        
        # Get full points for model (before subsample)
        full_points = points.copy()
        
        # Subsample points for display
        points = points[::self.subsample]
        
        # Plot new points
        self.scatter = self.ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                        c='cyan', s=0.1, alpha=0.5)
        
        # Draw ground truth boxes in RED (if enabled)
        if self.show_boxes:
            for box in boxes:
                self._draw_box(box, color='red')
        
        # Draw predicted boxes in GREEN (if detector available)
        pred_boxes = []
        if self.detector is not None:
            # Add intensity channel (zeros) for model
            if full_points.shape[1] == 3:
                full_points = np.hstack([full_points, np.zeros((len(full_points), 1))])
            pred_boxes = self.detector.predict(full_points, threshold=0.5)
            for pred in pred_boxes:
                cx, cy, cz, l, w, h, heading, score = pred
                box = {
                    'center': np.array([cx, -cy, cz]),  # flip y to match visualization
                    'size': np.array([l, w, h]),
                    'heading': -heading
                }
                self._draw_box(box, color='lime')
        
        # Update title
        title = f'Frame {self.current_frame + 1}/{self.total_frames}  |  GT: {len(boxes)} (red)'
        if self.detector:
            title += f'  |  Pred: {len(pred_boxes)} (green)'
        title += '  |  ←→ frames'
        
        self.fig.suptitle(title, color='white', fontsize=12)
        
        self.fig.canvas.draw_idle()
        msg = f"\rFrame {self.current_frame + 1}/{self.total_frames} | {len(points)} pts | GT: {len(boxes)}"
        if self.detector:
            msg += f" | Pred: {len(pred_boxes)}"
        print(msg, end='', flush=True)
    
    def _draw_box(self, box, color='red'):
        """Draw a 3D bounding box and store line references"""
        center = box['center']
        size = box['size']
        heading = box['heading']
        
        dx, dy, dz = size[0]/2, size[1]/2, size[2]/2
        
        corners = np.array([
            [-dx, -dy, -dz], [dx, -dy, -dz], [dx, dy, -dz], [-dx, dy, -dz],
            [-dx, -dy, dz], [dx, -dy, dz], [dx, dy, dz], [-dx, dy, dz]
        ])
        
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rot = np.array([[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]])
        corners = corners @ rot.T + center
        
        edges = [
            [0,1], [1,2], [2,3], [3,0],
            [4,5], [5,6], [6,7], [7,4],
            [0,4], [1,5], [2,6], [3,7]
        ]
        for e in edges:
            line, = self.ax.plot3D(*zip(corners[e[0]], corners[e[1]]), color=color, linewidth=1.5)
            if color == 'red':
                self.box_lines.append(line)
            else:
                self.pred_lines.append(line)
    
    def show(self):
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Waymo LiDAR Visualizer')
    parser.add_argument('--file', type=str, 
                        default='10203656353524179475_7625_000_7645_000.parquet',
                        help='Parquet filename to visualize')
    parser.add_argument('--frame', type=int, default=0, help='Starting frame index')
    parser.add_argument('--subsample', type=int, default=10, 
                        help='Show every Nth point (higher = faster)')
    parser.add_argument('--local', action='store_true', help='Use local data')
    parser.add_argument('--no-boxes', action='store_true', help='Hide bounding boxes')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for predictions')
    args = parser.parse_args()
    
    # Data paths - use absolute path based on script location
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    if args.local:
        base_path = os.path.join(project_dir, 'data', 'validation')
        lidar_path = os.path.join(base_path, 'lidar', args.file)
        box_path = os.path.join(base_path, 'lidar_box', args.file)
    else:
        lidar_path = f'gs://waymo-lidar-data/validation/lidar/{args.file}'
        box_path = f'gs://waymo-lidar-data/validation/lidar_box/{args.file}'
    
    print(f"Loading {lidar_path}...")
    lidar_df = pd.read_parquet(lidar_path)
    box_df = pd.read_parquet(box_path)
    
    print(f"Loaded {len(lidar_df['key.frame_timestamp_micros'].unique())} frames")
    
    # Load detector if checkpoint provided
    detector = None
    if args.checkpoint:
        import torch
        from inference import Detector
        
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        
        detector = Detector(args.checkpoint, device)
        print(f"\nModel loaded - will show predictions in GREEN")
    
    print("\nControls:")
    print("  ← →    : Previous/Next frame")
    print("  ↑ ↓    : Jump 10 frames")
    print("  + / -  : Zoom in/out")
    print("  Scroll : Zoom in/out")
    print("  R      : Reset view")
    print("  Q      : Quit")
    print("  Mouse  : Rotate view")
    if detector:
        print("\nVisualization:")
        print("  RED boxes   = Ground truth")
        print("  GREEN boxes = Model predictions")
    print()
    
    viewer = LiDARViewer(lidar_df, box_df, subsample=args.subsample, 
                         show_boxes=not args.no_boxes, detector=detector)
    viewer.current_frame = args.frame
    viewer.show()


if __name__ == "__main__":
    main()
