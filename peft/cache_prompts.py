import os
from typing import List, Union

import numpy as np
import torch
from chromadb import Collection, PersistentClient
from tqdm import tqdm

from models.Point_MAE_PEFT import Point_MAE

# class PointMAEEmbeddingFunction(EmbeddingFunction):
#     def __init__(self, model: Point_MAE): self.model = model
    
#     def __call__(self, input: Documents) -> List[List[float]]:
#         # Process input through model to get embeddings
#         with torch.no_grad():
#             embeddings = self.model.encode_pts(input)
        
#         # Convert torch tensor to list of numpy arrays
#         if isinstance(embeddings, torch.Tensor):
#             return embeddings.cpu().numpy().tolist()
#         return embeddings

def cache_encoded_data(
    dataloader, 
    model_path, 
    output_path, 
    device='cuda',
    collection_name='encoded_data'
):
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
    client = PersistentClient(path=output_path)
    collection = client.get_or_create_collection(collection_name)
    
    # Encode each batch and add to ChromaDB
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Encoding data")):
            # Assuming batch contains point cloud data
            # Modify according to your dataloader format
            points = batch.to(device)
            
            # Encode points
            encoded = model.encode_pts(points)
            
            # Convert to numpy and add to ChromaDB
            embeddings = encoded.cpu().numpy()
            ids = [f"point_cloud_{i}_{j}" for j in range(len(embeddings))]
            
            # Add embeddings to collection
            collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=[{"batch_idx": i, "point_idx": j} for j in range(len(embeddings))]
            )
    
    print(f"Saved encoded data to ChromaDB collection at {output_path}")


def load_cached_data(cache_path, collection_name='encoded_data') -> Collection:
    """
    Load cached encoded data from a ChromaDB collection.
    
    Args:
        cache_path: Path to the ChromaDB persistent storage directory
        collection_name: Name of the collection to load (default: 'encoded_data')
        
    Returns:
        collection: ChromaDB collection object containing the encoded data
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"ChromaDB directory not found at {cache_path}")
    
    # Load collection from ChromaDB
    client = PersistentClient(path=cache_path)
    try:
        collection = client.get_collection(collection_name)
        # Get collection info to print some statistics
        count = collection.count()
        print(f"Loaded ChromaDB collection '{collection_name}' with {count} entries")
        return collection
    except ValueError:
        raise ValueError(f"Collection '{collection_name}' not found in ChromaDB at {cache_path}")

def main():
    """CLI function to cache prompt embeddings from a dataset into ChromaDB"""
    import argparse
    import os
    import torch
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser(description='Cache prompt embeddings from dataset')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to pretrained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to dataset')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Output directory for ChromaDB storage')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for encoding')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use for encoding (cuda/cpu)')
    
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)

    # Load model
    model = Point_MAE_PEFT.load_from_checkpoint(args.model_path)
    model = model.to(args.device)
    model.eval()

    # Initialize dataset and dataloader
    # Note: Modify this according to your specific dataset class
    dataset = YourDataset(args.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Initialize ChromaDB
    client = PersistentClient(path=args.output_path)
    collection = client.create_collection(
        name="encoded_data",
        metadata={"description": "Cached prompt embeddings"}
    )

    # Cache the embeddings
    cache_embeddings(model, dataloader, collection, args.device, args.output_path)

if __name__ == "__main__":
    main()
