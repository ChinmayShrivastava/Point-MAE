import sqlite3

import numpy as np
import torch
from torch.utils.data import Dataset


class DrivAerNetSQLDataset(Dataset):
    """Dataset class for loading point clouds from SQLite database"""
    def __init__(
        self, 
        db_path: str,
        num_points: int = 8192
    ):
        """
        Args:
            db_path (str): Path to SQLite database containing point clouds
        """
        self.db_path = db_path
        self.num_points = num_points
        
        # Store only the count of available IDs
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM point_clouds")
            self.dataset_size = cursor.fetchone()[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # Load point cloud from database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM point_clouds WHERE id=?", (idx,))
            result = cursor.fetchone()
            
            if result is None:
                # Fallback: get the first available point cloud instead of raising an error
                print(f"Warning: No data found for index {idx}, using fallback data")
                cursor.execute("SELECT data FROM point_clouds LIMIT 1")
                result = cursor.fetchone()
                if result is None:
                    raise RuntimeError("Database appears to be empty")
                    
            binary_data = result[0]
            
        # Convert binary data to tensor (already normalized)
        point_cloud_array = np.frombuffer(binary_data, dtype=np.float32).reshape(8192, 4)
        # Make array writable and convert to correct shape
        point_cloud_array = point_cloud_array.copy()  # Make writable
        
        # # Sample num_points if less than total available points
        # if self.num_points < 8192:
        #     # Randomly sample indices without replacement
        #     sample_indices = np.random.choice(8192, self.num_points, replace=False)
        #     point_cloud_array = point_cloud_array[sample_indices]
        
        point_cloud = torch.from_numpy(point_cloud_array).permute(1, 0)  # Shape: (4, num_points)
        
        return point_cloud