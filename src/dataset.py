"""
Dataset for Waymo LiDAR object detection.
Loads point clouds and bounding box labels.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class WaymoDataset(Dataset):
    def __init__(self, data_dir, split='validation', max_points=100000):
        """
        Args:
            data_dir: Path to data directory containing lidar/ and lidar_box/ folders
            split: 'training' or 'validation'
            max_points: Maximum points to keep per frame
        """
        self.data_dir = data_dir
        self.split = split
        self.max_points = max_points
        
        # Find all parquet files
        lidar_dir = os.path.join(data_dir, split, 'lidar')
        self.files = [f for f in os.listdir(lidar_dir) if f.endswith('.parquet')]
        
        # Build index: (file_idx, timestamp)
        print(f"Building dataset index from {len(self.files)} files...")
        self.index = []
        for file_idx, filename in enumerate(self.files):
            lidar_path = os.path.join(lidar_dir, filename)
            df = pd.read_parquet(lidar_path, columns=['key.frame_timestamp_micros'])
            timestamps = df['key.frame_timestamp_micros'].unique()
            for ts in timestamps:
                self.index.append((file_idx, ts))
        
        print(f"Total frames: {len(self.index)}")
    
    def __len__(self):
        return len(self.index)
    
    def _range_image_to_points(self, range_image, laser_name):
        """Convert range image to point cloud [N, 4] with x,y,z,intensity"""
        height, width = range_image.shape[0], range_image.shape[1]
        range_values = range_image[:, :, 0]
        intensity = range_image[:, :, 1]
        valid_mask = range_values > 0
        
        if np.sum(valid_mask) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        
        # Vertical angles based on sensor
        if laser_name == 0:  # TOP
            inclination = np.linspace(-np.pi/6, np.pi/6, height)
        else:
            inclination = np.linspace(-np.pi/8, np.pi/8, height)
        
        azimuth = np.linspace(-np.pi, np.pi, width)
        azimuth_grid, inclination_grid = np.meshgrid(azimuth, inclination)
        
        # Spherical to Cartesian
        x = range_values * np.cos(inclination_grid) * np.cos(azimuth_grid)
        y = range_values * np.cos(inclination_grid) * np.sin(azimuth_grid)
        z = range_values * np.sin(inclination_grid)
        
        points = np.stack([x, y, z, intensity], axis=-1)
        return points[valid_mask].astype(np.float32)
    
    def __getitem__(self, idx):
        file_idx, timestamp = self.index[idx]
        filename = self.files[file_idx]
        
        # Load LiDAR data
        lidar_path = os.path.join(self.data_dir, self.split, 'lidar', filename)
        lidar_df = pd.read_parquet(lidar_path)
        frame_data = lidar_df[lidar_df['key.frame_timestamp_micros'] == timestamp]
        
        # Convert all sensors to point cloud
        all_points = []
        for _, row in frame_data.iterrows():
            range_bytes = row['[LiDARComponent].range_image_return1.values']
            range_shape = row['[LiDARComponent].range_image_return1.shape']
            
            range_image = np.frombuffer(range_bytes, dtype=np.float32)
            shape = tuple(range_shape) if isinstance(range_shape, (list, tuple)) else range_shape
            range_image = range_image.reshape(shape)
            
            laser_name = row['key.laser_name']
            points = self._range_image_to_points(range_image, laser_name)
            if len(points) > 0:
                all_points.append(points)
        
        points = np.vstack(all_points) if all_points else np.zeros((0, 4), dtype=np.float32)
        
        # Subsample if too many points
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
        
        # Load bounding boxes
        box_path = os.path.join(self.data_dir, self.split, 'lidar_box', filename)
        box_df = pd.read_parquet(box_path)
        box_frame = box_df[box_df['key.frame_timestamp_micros'] == timestamp]
        
        boxes = []
        box_center_col = '[LiDARBoxComponent].box.center.x'
        if box_center_col in box_frame.columns:
            box_rows = box_frame[box_frame[box_center_col].notna()]
            for _, row in box_rows.iterrows():
                obj_type = row['[LiDARBoxComponent].type']
                if obj_type == 1:  # Vehicle
                    boxes.append([
                        row['[LiDARBoxComponent].box.center.x'],
                        row['[LiDARBoxComponent].box.center.y'],
                        row['[LiDARBoxComponent].box.center.z'],
                        row['[LiDARBoxComponent].box.size.x'],  # length
                        row['[LiDARBoxComponent].box.size.y'],  # width
                        row['[LiDARBoxComponent].box.size.z'],  # height
                        row['[LiDARBoxComponent].box.heading'],
                    ])
        
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 7), dtype=np.float32)
        
        return {
            'points': torch.from_numpy(points),
            'boxes': torch.from_numpy(boxes),
            'num_boxes': len(boxes)
        }


def collate_fn(batch):
    """Custom collate for variable-sized data"""
    return {
        'points': [item['points'] for item in batch],
        'boxes': [item['boxes'] for item in batch],
        'num_boxes': [item['num_boxes'] for item in batch]
    }

