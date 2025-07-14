"""
helper functions for V-JEPA 2 frame decoder.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt


def calculate_psnr(img1, img2, max_value=1.0):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Arguments:
        img1: First image (tensor or numpy array)
        img2: Second image (tensor or numpy array)
        max_value: Maximum value of the images (default: 1.0 for normalized images)
        
    Returns:
        PSNR value
    """
    # Convert tensors to numpy arrays
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
    
    # Calculate MSE
    mse = mean_squared_error(img1.flatten(), img2.flatten())
    if mse == 0:
        return float('inf')
        
    # Calculate PSNR
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index between two images.
    
    Arguments:
        img1: First image (tensor or numpy array)
        img2: Second image (tensor or numpy array)
        
    Returns:
        SSIM value
    """
    # Convert tensors to numpy arrays
    if torch.is_tensor(img1):
        img1 = img1.permute(1, 2, 0).detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.permute(1, 2, 0).detach().cpu().numpy()
    
    # Ensure the images are in the same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Images have different shapes: {img1.shape} vs {img2.shape}")
    
    # Calculate SSIM
    ssim = structural_similarity(img1, img2, channel_axis=2, data_range=1.0)
    return ssim


def load_image(path, size=224):
    """
    Load an image from a file path and preprocess it.
    
    Arguments:
        path: Path to the image file
        size: Size to resize the image to
        
    Returns:
        Preprocessed image tensor
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(path).convert('RGB')
    image = transform(image)
    return image


def save_comparison_grid(original, reconstructed, output_path):
    """
    Save a grid of original and reconstructed images for comparison.
    
    Arguments:
        original: Original images (B, C, H, W)
        reconstructed: Reconstructed images (B, C, H, W)
        output_path: Path to save the grid
        
    Returns:
        None
    """
    batch_size = original.shape[0]
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))
    
    # If batch size is 1, ensure axes is still a 2D array
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Original image
        orig_img = original[i].permute(1, 2, 0).cpu().numpy()
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')
        
        # Reconstructed image
        recon_img = reconstructed[i].permute(1, 2, 0).cpu().numpy()
        axes[i, 1].imshow(recon_img)
        
        # Calculate metrics
        psnr = calculate_psnr(orig_img, recon_img)
        ssim = calculate_ssim(orig_img, recon_img)
        axes[i, 1].set_title(f"Reconstructed (PSNR: {psnr:.2f}, SSIM: {ssim:.4f})")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def create_run_directory(base_dir="./runs"):
    """
    Create a unique run directory for this training run.
    
    Arguments:
        base_dir: Base directory for runs
        
    Returns:
        Path to the created directory
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Find the next run number
    existing_runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")]
    run_numbers = [int(d.split("_")[1]) for d in existing_runs if d.split("_")[1].isdigit()]
    next_run = max(run_numbers) + 1 if run_numbers else 0
    
    # Create the run directory
    run_dir = os.path.join(base_dir, f"run_{next_run:03d}")
    os.makedirs(run_dir, exist_ok=True)
    
    return run_dir


def get_device():
    """
    Get the appropriate device for training.
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("CUDA is not available. Using CPU instead.")
        return torch.device("cpu")