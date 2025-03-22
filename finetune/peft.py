import enum
import os

import torch
import torch.nn as nn
import wandb

from datasets.DrivaernetDataset import get_dataloaders
from models.Point_MAE_PEFT import PointTransformer_PEFT
from peft.cache_prompts import load_encoder_model, load_cached_data


class LossType(enum.Enum):
    MSE = "mse"
    L1 = "l1"
    L2 = "l2"

config = {
    "batch_size": 16,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "models/Point_MAE_PEFT.py",
    "output_path": "data/",
    "data_path": "data/drivaernet_db.sqlite",
    "wandb_project": "drivaernet-peft",
    "wandb_entity": "drivaernet",
    "wandb_name": "point-mae-peft",
    "num_epochs": 10,
    "loss_type": LossType.MSE,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "k": 10,
    "prompt_bank": load_cached_data(),
    "prompt_encoder": load_encoder_model(),
    "early_stop_patience": 5,
    "checkpoint_every": 1
}

def get_loss_fn(loss_type: LossType):
    if loss_type == LossType.MSE:
        return nn.MSELoss()
    elif loss_type == LossType.L1:
        return nn.L1Loss()
    elif loss_type == LossType.L2:
        return nn.MSELoss()
    
def get_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    
def get_scheduler(optimizer: torch.optim.Optimizer, num_epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

def process_batch(batch):
    points = batch["points"]
    return points[:, :, :3], points[:, :, 3]

def finetune_point_mae_peft(
    config: dict
):
    train_loader, val_loader, test_loader = get_dataloaders(
        db_path=config["data_path"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )
    model = PointTransformer_PEFT(
        config=config, 
        embedding_model=load_encoder_model(),
        k=config["k"],
        prompt_bank=config["prompt_bank"],
        prompt_encoder=config["prompt_encoder"]
    )
    model = model.to(config["device"])
    model.prepare_for_peft()

    wandb.init(
        project=config["wandb_project"],
        entity=config["wandb_entity"],
        name=config["wandb_name"]
    )
    
    # print the total number of parameters and the trainable parameters
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    optimizer = get_optimizer(model, config["lr"], config["weight_decay"])
    max_grad_norm = config.get("max_grad_norm", 1.0)
    loss_fn = get_loss_fn(config["loss_type"])
    scheduler = get_scheduler(optimizer, config["num_epochs"])

    # Create checkpoint directory
    checkpoint_dir = os.path.join(config["output_path"]+f"/{config['wandb_name']}", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')
    early_stop_patience = config["early_stop_patience"]
    epochs_no_improve = 0

    for epoch in range(config["num_epochs"]):
        # Training
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Get data and move to device
            pts, target = process_batch(batch)
            pts = pts.to(config["device"])
            target = target.to(config["device"])
            
            # Forward pass
            output = model(pts)
            loss = loss_fn(output, target)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(pts)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        # Step the scheduler
        scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                pts, target = process_batch(batch)
                pts = pts.to(config["device"])
                target = target.to(config["device"])
                
                output = model(pts)
                loss = loss_fn(output, target)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Test 
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                pts, target = process_batch(batch)
                pts = pts.to(config["device"]) 
                target = target.to(config["device"])
                
                output = model(pts)
                loss = loss_fn(output, target)
                total_test_loss += loss.item()
                
        avg_test_loss = total_test_loss / len(test_loader)
        
        # Save periodic checkpoint
        if epoch % config.get("checkpoint_every", 1) == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'test_loss': avg_test_loss,
            }, checkpoint_path)
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "test_loss": avg_test_loss,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        
        print(f'Epoch {epoch} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Test Loss: {avg_test_loss:.6f}')
        
    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_model.pth"))
    wandb.finish()
    return model

if __name__ == "__main__":
    finetune_point_mae_peft(config)