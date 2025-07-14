"""
Frame decoder model for V-JEPA 2 representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange

from .config import get_model_dim


class UpsamplingBlock(nn.Module):
    """
    Upsampling block for the decoder, consisting of transpose convolution followed by
    regular convolution with normalization and activation.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            # Transpose convolution for upsampling
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.GroupNorm(32 if out_channels >= 32 else out_channels, out_channels),
            nn.GELU(),
            # Regular convolution for refining features
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=3, padding=1
            ),
            nn.GroupNorm(32 if out_channels >= 32 else out_channels, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout_rate)
        )
        
    def forward(self, x):
        return self.layers(x)


class FrameDecoder(nn.Module):
    """
    Frame decoder that maps V-JEPA 2 patch embeddings to RGB images.
    
    Arguments:
        embedding_dim: Dimension of the input embeddings from V-JEPA 2
        patch_size: Size of the patches used in V-JEPA 2 (default: 16)
        image_size: Size of the output image (default: 224)
        channels: Number of output channels (default: 3 for RGB)
        dropout_rate: Dropout rate for regularization (default: 0.1)
    """
    def __init__(self, embedding_dim, patch_size=16, image_size=224, channels=3, dropout_rate=0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.channels = channels
        
        # Compute the number of patches in each dimension
        self.num_patches_per_dim = image_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 2
        
        # Initial projection to adapt dimensions if needed
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Define upsampling blocks
        self.upsampling = nn.ModuleList([
            # Progressive upsampling blocks with decreasing feature dimensions
            UpsamplingBlock(embedding_dim, 512, dropout_rate),  # 14x14 -> 28x28
            UpsamplingBlock(512, 256, dropout_rate),            # 28x28 -> 56x56
            UpsamplingBlock(256, 128, dropout_rate),           # 56x56 -> 112x112
            UpsamplingBlock(128, 64, dropout_rate),            # 112x112 -> 224x224
        ])
        
        # Final layer to produce RGB image with pixel values in [0,1] range
        self.to_rgb = nn.Sequential(
            nn.Conv2d(64, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Forward pass of the decoder.
        
        Arguments:
            x: Input tensor of shape (batch_size, num_patches, embedding_dim)
        
        Returns:
            Tensor of shape (batch_size, channels, image_size, image_size)
        """
        batch_size = x.shape[0]
        
        # Project embeddings
        x = self.projection(x)
        
        # Reshape from (batch_size, num_patches, embedding_dim) to 
        # (batch_size, embedding_dim, sqrt(num_patches), sqrt(num_patches))
        x = rearrange(
            x, 
            'b (h w) c -> b c h w', 
            h=self.num_patches_per_dim, 
            w=self.num_patches_per_dim
        )
        
        # Apply upsampling blocks
        for block in self.upsampling:
            x = block(x)
            
        # Convert to RGB
        x = self.to_rgb(x)
        
        return x


def create_frame_decoder(model_type="vit_large", patch_size=16, image_size=224):
    """
    Factory function to create a frame decoder based on the V-JEPA 2 model type.
    
    Arguments:
        model_type: V-JEPA 2 model type (default: "vit_large")
        patch_size: Size of the patches used in V-JEPA 2 (default: 16)
        image_size: Size of the output image (default: 224)
    
    Returns:
        FrameDecoder instance
    """
    embedding_dim = get_model_dim(model_type)
    
    return FrameDecoder(
        embedding_dim=embedding_dim,
        patch_size=patch_size,
        image_size=image_size
    )


if __name__ == "__main__":
    # Test the frame decoder
    batch_size = 2
    model_type = "vit_large"
    embedding_dim = 1024
    patch_size = 16
    image_size = 224
    num_patches = (image_size // patch_size) ** 2
    
    # Create random embeddings
    embeddings = torch.randn(batch_size, num_patches, embedding_dim)
    
    # Create frame decoder
    decoder = create_frame_decoder(model_type, patch_size, image_size)
    
    # Forward pass
    output = decoder(embeddings)
    
    # Check output shape
    expected_shape = (batch_size, 3, image_size, image_size)
    assert output.shape == expected_shape, f"Output shape {output.shape} doesn't match expected shape {expected_shape}"
    
    print(f"Frame decoder test passed! Output shape: {output.shape}")