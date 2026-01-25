"""
Dataset for Waymo LiDAR object detection.
Loads point clouds and bounding box labels from GCS or local.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os



class WaymoDataset(Dataset):
    def __init__(self, data_dir, split='training', max_points=100000, use_gcs=False, max_files=None):
        """
        Args:
            data_dir: Path to data directory (ignored if use_gcs=True)
            split: 'training' or 'validation'
            max_points: Maximum points to keep per frame
            use_gcs: If True, stream from GCS bucket
            max_files: Limit number of files to use (None = all)
        """
        self.split = split
        self.max_points = max_points
        self.use_gcs = use_gcs

        if use_gcs:
            import gcsfs
            # If you are using a PUBLIC bucket (like the official one)
            # anon=True is correct. If you use your own private bucket,
            # drop anon=True and use credentials instead.
            self.fs = gcsfs.GCSFileSystem(anon=True)

            # 👇 CHANGE THIS if your bucket name is different
            # e.g. if your data is in "waymo-lidar-data", set that here:
            # self.bucket = "waymo-lidar-data"
            self.bucket = "waymo-lidar-data"

            # gcsfs paths do NOT include "gs://"
            self.lidar_dir = f"{self.bucket}/{split}/lidar"
            self.box_dir = f"{self.bucket}/{split}/lidar_box"

            print(f"Listing files from gs://{self.lidar_dir} ...")
            all_files = self.fs.ls(self.lidar_dir)
            # all_files look like: "waymo_open_dataset_v_2_0_1/training/lidar/xxx.parquet"
            self.files = sorted([f for f in all_files if f.endswith('.parquet')])

            if not self.files:
                raise RuntimeError(f"No parquet files found in gs://{self.lidar_dir}")
        else:
            self.fs = None
            self.data_dir = data_dir
            lidar_dir = os.path.join(data_dir, split, 'lidar')
            self.files = [os.path.join(lidar_dir, f)
                          for f in os.listdir(lidar_dir)
                          if f.endswith('.parquet')]

        # Limit files if specified
        if max_files:
            self.files = self.files[:max_files]

        # Build index: (file_idx, timestamp)
        print(f"Building dataset index from {len(self.files)} files...")
        self.index = []
        for file_idx, filepath in enumerate(self.files):
            if self.use_gcs:
                # filepath is like "waymo_open_dataset_v_2_0_1/training/lidar/xxx.parquet"
                lidar_path = f"gs://{filepath}"
                df = pd.read_parquet(
                    lidar_path,
                    columns=['key.frame_timestamp_micros'],
                    storage_options={"token": "anon"}
                )
            else:
                df = pd.read_parquet(filepath, columns=['key.frame_timestamp_micros'])

            timestamps = df['key.frame_timestamp_micros'].unique()
            for ts in timestamps:
                self.index.append((file_idx, ts))

            if (file_idx + 1) % 10 == 0:
                print(f"  Indexed {file_idx + 1}/{len(self.files)} files...")

        print(f"Total frames: {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def _get_paths(self, file_idx):
        """Get lidar and box paths for a file."""
        if self.use_gcs:
            # self.files[file_idx] = "bucket/split/lidar/xxx.parquet"
            lidar_path = f"gs://{self.files[file_idx]}"
            filename = self.files[file_idx].split('/')[-1]
            box_path = f"gs://{self.box_dir}/{filename}"
        else:
            lidar_path = self.files[file_idx]
            filename = os.path.basename(lidar_path)
            box_path = os.path.join(self.data_dir, self.split, 'lidar_box', filename)
        return lidar_path, box_path

    # keep your _range_image_to_points and __getitem__ methods as-is,
    # but with a small tweak to GCS reads:

    def __getitem__(self, idx):
        file_idx, timestamp = self.index[idx]
        lidar_path, box_path = self._get_paths(file_idx)

        # Load LiDAR data
        if self.use_gcs:
            lidar_df = pd.read_parquet(
                lidar_path,
                storage_options={"token": "anon"}
            )
        else:
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

        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]

        # Load bounding boxes
        if self.use_gcs:
            box_df = pd.read_parquet(
                box_path,
                storage_options={"token": "anon"}
            )
        else:
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
                        row['[LiDARBoxComponent].box.size.x'],
                        row['[LiDARBoxComponent].box.size.y'],
                        row['[LiDARBoxComponent].box.size.z'],
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