import sqlite3

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import pandas as pd

class DrivAerNetSQLDataset(Dataset):
    """Dataset class for loading point clouds from SQLite database"""
    def __init__(
        self, 
        db_path: str,
        num_points: int = 1024
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
            cursor.execute("SELECT id, data FROM point_clouds WHERE id=?", (idx,))
            result = cursor.fetchone()
            
            if result is None:
                # Fallback: get the first available point cloud instead of raising an error
                print(f"Warning: No data found for index {idx}, using fallback data")
                cursor.execute("SELECT id, data FROM point_clouds LIMIT 1")
                result = cursor.fetchone()
                if result is None:
                    raise RuntimeError("Database appears to be empty")
                    
            point_id, binary_data = result
            
        # Convert binary data to tensor (already normalized)
        point_cloud_array = np.frombuffer(binary_data, dtype=np.float32).reshape(8192, 4)
        # Make array writable and convert to correct shape
        point_cloud_array = point_cloud_array.copy()  # Make writable
        
        # Sample num_points if less than total available points
        if self.num_points < 8192:
            # Randomly sample indices without replacement
            sample_indices = np.random.choice(8192, self.num_points, replace=False)
            point_cloud_array = point_cloud_array[sample_indices]
        
        return {
            'id': point_id,
            'points': point_cloud_array  # Shape: (num_points, 4)
        }
    
def get_dataloaders(
    db_path: str,
    batch_size: int,
    num_workers: int
) -> tuple:
    """
    Prepare and return the training, validation, and test DataLoader objects.

    Args:
        db_path (str): Path to SQLite database containing point clouds
        batch_size (int): The number of samples per batch to load
        num_workers (int): Number of worker processes for data loading

    Returns:
        tuple: A tuple containing the training DataLoader, validation DataLoader, and test DataLoader
    """
    full_dataset = DrivAerNetSQLDataset(db_path=db_path)
    
    train_ids = pd.read_csv('train_val_test_splits/train_design_ids.txt', header=None).values.flatten()
    val_ids = pd.read_csv('train_val_test_splits/val_design_ids.txt', header=None).values.flatten()
    test_ids = pd.read_csv('train_val_test_splits/test_design_ids.txt', header=None).values.flatten()
    
    train_dataset = Subset(full_dataset, train_ids)
    val_dataset = Subset(full_dataset, val_ids)
    test_dataset = Subset(full_dataset, test_ids)
    
    def worker_init_fn(worker_id):
        torch.cuda.empty_cache()
    
    # Optimize DataLoader settings for memory efficiency
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': False,
        'persistent_workers': False,
        'prefetch_factor': 1,          # Reduced from 2 to 1
        'drop_last': True,
        'worker_init_fn': worker_init_fn  # Add worker initialization function
    }

        
    # Reduce batch size if needed
    if batch_size > 8:  # Adjust this threshold based on your GPU
        print(f"Warning: Large batch size ({batch_size}) may cause OOM errors. Consider reducing it.")
    
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        **dataloader_kwargs
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        shuffle=False, 
        **dataloader_kwargs
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        shuffle=False, 
        **dataloader_kwargs
    )
    
    return train_dataloader, val_dataloader, test_dataloader