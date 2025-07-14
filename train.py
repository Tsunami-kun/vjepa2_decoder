"""
Training code for the V-JEPA 2 frame decoder.
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import torch.utils.tensorboard as tensorboard

from .config import DEFAULT_CONFIG

# Local imports
from .model import create_frame_decoder
from .dataset import LiberoImageDataset, create_frame_decoder_datasets


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, log_interval=DEFAULT_CONFIG['training']['log_interval']):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, (embeddings, images, _) in enumerate(train_loader):
        embeddings, images = embeddings.to(device), images.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(embeddings)
        loss = criterion(output, images)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        
        # Log progress
        if batch_idx % log_interval == 0:
            elapsed = time.time() - start_time
            print(f'Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\t'
                  f'Time: {elapsed:.2f}s')
            start_time = time.time()
    
    # Return average loss
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, epoch, output_dir=None):
    """Evaluate the model on the validation set"""
    model.eval()
    val_loss = 0
    mse_loss = nn.MSELoss()
    psnr_sum = 0
    
    with torch.no_grad():
        for embeddings, images, paths in val_loader:
            embeddings, images = embeddings.to(device), images.to(device)
            
            # Forward pass
            output = model(embeddings)
            loss = criterion(output, images)
            val_loss += loss.item()
            
            # Calculate PSNR
            mse = mse_loss(output, images)
            psnr = 10 * torch.log10(1 / mse)
            psnr_sum += psnr.item()
            
            # Save sample reconstructions
            if output_dir and epoch % 5 == 0:
                os.makedirs(output_dir, exist_ok=True)
                # Select first 8 images in batch
                num_samples = min(8, len(output))
                samples = torch.cat([
                    images[:num_samples],
                    output[:num_samples],
                ], dim=0)
                
                # Create grid
                grid = make_grid(samples, nrow=num_samples, normalize=True)
                save_image(grid, os.path.join(output_dir, f'epoch_{epoch}_samples.png'))
                break  # Save only the first batch
    
    # Return average loss and PSNR
    avg_val_loss = val_loss / len(val_loader)
    avg_psnr = psnr_sum / len(val_loader)
    
    print(f'Validation: Loss: {avg_val_loss:.4f}, PSNR: {avg_psnr:.2f}')
    return avg_val_loss, avg_psnr


def train(
    model, 
    train_dataset, 
    val_dataset, 
    output_dir, 
    epochs=DEFAULT_CONFIG['training']['epochs'], 
    batch_size=DEFAULT_CONFIG['training']['batch_size'], 
    lr=DEFAULT_CONFIG['training']['learning_rate'], 
    weight_decay=DEFAULT_CONFIG['training']['weight_decay'],
    device="cuda",
    num_workers=4,
):
    """
    Train the frame decoder model.
    
    Arguments:
        model: Frame decoder model
        train_dataset: Training dataset
        val_dataset: Validation dataset
        output_dir: Directory to save checkpoints and logs
        epochs: Number of epochs to train
        batch_size: Batch size for training
        lr: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to train on
        num_workers: Number of workers for data loading
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    # TensorBoard writer
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(log_dir=log_dir)
    
    # Move model to device
    model = model.to(device)
    
    # Track best model
    best_val_loss = float('inf')
    samples_dir = os.path.join(output_dir, "samples")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss, val_psnr = validate(model, val_loader, criterion, device, epoch, samples_dir)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('PSNR/validation', val_psnr, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save model checkpoint
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_psnr': val_psnr,
        }, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(output_dir, 'best_model.pt')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    writer.close()
    print("Training completed!")


def setup_model_and_train(
    libero_image_dir, 
    encoder,
    model_type="vit_large", 
    output_dir="./output",
    epochs=50, 
    batch_size=64, 
    lr=1e-4, 
    weight_decay=0.01,
    device="cuda",
    num_workers=4,
    val_split=0.1,
    seed=42,
):
    """
    Setup the model and train it.
    
    Arguments:
        libero_image_dir: Directory containing Libero images
        encoder: V-JEPA 2 encoder model
        model_type: V-JEPA 2 model type
        output_dir: Directory to save checkpoints and logs
        epochs: Number of epochs to train
        batch_size: Batch size for training
        lr: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to train on
        num_workers: Number of workers for data loading
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
    """
    # Create frame decoder datasets
    train_dataset, val_dataset = create_frame_decoder_datasets(
        libero_image_dir=libero_image_dir,
        encoder=encoder,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        val_split=val_split,
        seed=seed,
    )
    
    # Create frame decoder model
    model = create_frame_decoder(
        model_type=model_type,
        patch_size=16,
        image_size=224,
    )
    
    # Train the model
    train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train V-JEPA 2 frame decoder")
    parser.add_argument("--libero-dir", type=str, default="/home/tsunami/vjepa2/libero/libero_images",
                        help="Directory containing Libero images")
    parser.add_argument("--checkpoint", type=str, default="/home/tsunami/vjepa2/ckpts/vitl.pt",
                        help="Path to V-JEPA 2 checkpoint")
    parser.add_argument("--model-type", type=str, default="vit_large",
                        choices=["vit_large", "vit_huge", "vit_giant", "vit_giant_384"],
                        help="V-JEPA 2 model type")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay for regularization")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to train on")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print(f"Frame decoder training code ready. To run, provide a proper V-JEPA 2 encoder checkpoint.")