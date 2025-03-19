import os

import torch
from tqdm import tqdm

from models.Point_MAE_PEFT import Point_MAE


def cache_encoded_data(dataloader, model_path, output_path, device='cuda'):
    """
    Load pretrained PointTransformer model and encode data from dataloader.
    Save encoded representations to a .pt file.
    
    Args:
        dataloader: torch.utils.data.DataLoader instance
        model_path: Path to pretrained model checkpoint
        output_path: Path to save encoded data
        device: Device to run model on
    """
    # Load pretrained model
    model = Point_MAE(config=None)  # Config will be loaded from checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    # Create storage for encoded data
    encoded_data = []
    
    # Encode each batch
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding data"):
            # Assuming batch contains point cloud data
            # Modify according to your dataloader format
            points = batch.to(device)
            
            # Encode points
            encoded = model.encode_pts(points)
            encoded_data.append(encoded.cpu())
            
    # Concatenate all batches
    encoded_data = torch.cat(encoded_data, dim=0)
    
    # Save encoded data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(encoded_data, output_path)
    print(f"Saved encoded data to {output_path}")


def load_cached_data(cache_path):
    """
    Load cached encoded data into a tensor of shape [N, D].
    
    Args:
        cache_path: Path to the cached data file (.pt)
        
    Returns:
        torch.Tensor: Loaded data with shape [N, D] where N is the total number
                     of elements and D is the encoding dimension
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found at {cache_path}")
    
    # Load cached data
    cached_data = torch.load(cache_path)
    
    # Ensure proper shape
    if cached_data.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape: {cached_data.shape}")
    
    print(f"Loaded cached data with shape: {cached_data.shape}")
    return cached_data
