import torch

def cache_encoded_data(
    dataloader, 
    model_path, 
    output_path, 
    device='cuda',
    collection_name='encoded_data',
    config_path='cfgs/pretrain.yaml'
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
    import easydict
    import torch
    import yaml
    from chromadb import PersistentClient
    from tqdm import tqdm

    from models.Point_MAE import Point_MAE
    from tools import builder
    assert config_path is not None, "Config path is required"
    
    # Load config if provided
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = easydict.EasyDict(config['model'])  # Directly convert model config section to EasyDict
    
    if config is None:
        raise ValueError("Config is None")
    
    # Check if data is already cached
    client = PersistentClient(
        path=output_path
    )
    try:
        collection = client.get_collection(collection_name)
        count = collection.count()
        print(f"Found existing cached data with {count} entries at {output_path}")
        return collection
    except:
        # Collection doesn't exist, proceed with caching
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    # Load pretrained model
    model = Point_MAE(config=config)  # Config will be loaded from checkpoint
    builder.load_model(model, ckpt_path=model_path)
    # checkpoint = torch.load(model_path, map_location=device)
    # model.load_model_from_ckpt(model_path)
    
    model = model.to(device)
    model.eval()
    
    # Encode each batch and add to ChromaDB
    total_batches = len(dataloader)
    progress_bar = tqdm(total=total_batches, desc="Encoding data")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Assuming batch contains point cloud data
            # Modify according to your dataloader format
            
            ids = batch['id']
            points = batch['points'].to(device)
            
            # Encode points
            encoded = model.encode_pts(points[:, :, :3].contiguous()) # this takes in B N 3
            
            # Convert to numpy and add to ChromaDB
            if device == 'cuda':
                embeddings = encoded.cpu().detach().numpy()
            else:
                embeddings = encoded.numpy() # B C
            
            # Add embeddings to collection
            collection.add(
                embeddings=embeddings,
                ids=ids,
                metadatas=[{"id": id} for id in ids]
            )
            
            progress_bar.update(1)
            
    progress_bar.close()
    print(f"Saved encoded data to ChromaDB collection at {output_path}")
    
    return collection


def load_cached_data(cache_path, collection_name='encoded_data'):
    """
    Load cached encoded data from a ChromaDB collection.
    
    Args:
        cache_path: Path to the ChromaDB persistent storage directory
        collection_name: Name of the collection to load (default: 'encoded_data')
        
    Returns:
        collection: ChromaDB collection object containing the encoded data
    """
    from chromadb import PersistentClient
    import os

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

def load_encoder_model(model_path='data/weights/pretrain.pth', config_path='cfgs/pretrain.yaml', device='cuda'):
    """
    Load pretrained PointTransformer model with point_peft.
    
    Args:
        model_path: Path to pretrained model checkpoint
        config_path: Path to model config file
        device: Device to run model on ('cuda' or 'cpu')
        
    Returns:
        model: Loaded Point_MAE model on specified device
    """
    import easydict
    import yaml
    from models.Point_MAE import Point_MAE
    from tools import builder

    # Load config if provided
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = easydict.EasyDict(config['model'])

    if config is None:
        raise ValueError("Config is None")

    # Load pretrained model
    model = Point_MAE(config=config)
    builder.load_model(model, ckpt_path=model_path)
    
    model = model.to(device)
    model.eval()
    
    return model

def main(
    model_path: str,
    data_path: str,
    output_path: str,
    batch_size: int,
    num_workers: int,
    device: torch.device
):
    """CLI function to cache prompt embeddings from a dataset into ChromaDB"""
    import os
    from datasets.DrivaernetDataset import get_dataloaders
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Initialize dataset and dataloader
    # Note: Modify this according to your specific dataset class
    train_dataloader, _, _ = get_dataloaders(
        db_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Cache the embeddings
    cache_encoded_data(
        train_dataloader, 
        model_path, 
        output_path, 
        device, 
        "encoded_data"
    )

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='Cache prompt embeddings from dataset')
#     parser.add_argument('--model_path', type=str, required=True,
#                       help='Path to pretrained model checkpoint')
#     parser.add_argument('--data_path', type=str, required=True,
#                       help='Path to sqlite database')
#     parser.add_argument('--output_path', type=str, default='data/',
#                       help='Output directory for ChromaDB storage')
#     parser.add_argument('--batch_size', type=int, default=16,
#                       help='Batch size for encoding')
#     parser.add_argument('--num_workers', type=int, default=4,
#                       help='Number of workers for data loading')
#     parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#                       help='Device to use for encoding (cuda/cpu)')
#     args = parser.parse_args()
#     main(
#         args.model_path,
#         args.data_path,
#         args.output_path,
#         args.batch_size,
#         args.num_workers,
#         args.device
#     )
