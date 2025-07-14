"""
Dataset handling for training the V-JEPA 2 frame decoder.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import glob
from einops import rearrange
import warnings
import logging

from .config import get_transform, DEFAULT_CONFIG

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class LiberoImageDataset(Dataset):
    """
    Dataset for Libero images, returning the original images.
    """
    def __init__(self, image_dir, transform=None, filter_empty=True):
        """
        Initialize the dataset.
        
        Arguments:
            image_dir: Directory containing Libero images
            transform: Optional transform to apply to the images
            filter_empty: Whether to filter out empty image files
        """
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        # Filter out empty files if requested
        if filter_empty:
            print(f"Found {len(self.image_paths)} images, filtering empty files...")
            valid_paths = []
            for path in self.image_paths:
                if os.path.getsize(path) > 0:
                    valid_paths.append(path)
            
            self.image_paths = valid_paths
            print(f"After filtering: {len(self.image_paths)} valid images")
            
            if len(self.image_paths) == 0:
                raise ValueError(f"No valid (non-empty) images found in {image_dir}")
        
        # Default transform if none provided
        if transform is None:
            self.transform = get_transform()
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        max_retries = DEFAULT_CONFIG['data']['max_retries']
        
        for attempt in range(max_retries):
            try:
                image = Image.open(img_path).convert('RGB')
                
                # Apply transformations
                if self.transform:
                    image = self.transform(image)
                    
                return image, img_path
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Error loading image {img_path} after {max_retries} attempts: {e}")
                    # Instead of returning a blank image, raise an exception so the caller can handle it properly
                    raise ValueError(f"Failed to load image {img_path} after {max_retries} attempts") from e
                logger.warning(f"Attempt {attempt + 1} failed for {img_path}: {e}, retrying...")


class FrameDecoderDataset(Dataset):
    """
    Dataset that pairs V-JEPA 2 embeddings with original images for decoder training.
    """
    def __init__(self, embeddings, images, image_paths=None):
        """
        Initialize the dataset.
        
        Arguments:
            embeddings: Tensor of V-JEPA 2 embeddings (N, num_patches, embedding_dim)
            images: Tensor of original images (N, C, H, W)
            image_paths: Optional list of image paths for reference
        """
        assert embeddings.shape[0] == images.shape[0], "Embeddings and images must have the same batch dimension"
        self.embeddings = embeddings
        self.images = images
        self.image_paths = image_paths
        
    def __len__(self):
        return self.embeddings.shape[0]
    
    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        image = self.images[idx]
        
        if self.image_paths is not None:
            return embedding, image, self.image_paths[idx]
        else:
            return embedding, image


class StreamingFrameDecoderDataset(Dataset):
    """
    Memory-efficient dataset that loads embeddings and images in chunks.
    """
    def __init__(self, embeddings_path, chunk_size=None, split='all', config=None):
        """
        Initialize the dataset.
        
        Arguments:
            embeddings_path: Path to the directory containing embeddings and images
            chunk_size: Number of samples to load at once
            split: Dataset split ('train', 'val', or 'all')
            config: Optional configuration dictionary
        """
        self.embeddings_path = embeddings_path
        
        # Use configuration if provided
        if config is None:
            config = DEFAULT_CONFIG
        
        # Use chunk size from config if not specified
        if chunk_size is None:
            chunk_size = config['data']['streaming']['chunk_size']
        
        self.chunk_size = chunk_size
        self.split = split
        self.max_cache_chunks = config['data']['streaming']['max_cache_chunks']
        
        # Load metadata
        self.metadata_path = os.path.join(embeddings_path, "metadata.pt")
        self.embeddings_file = os.path.join(embeddings_path, "embeddings.pt")
        self.images_file = os.path.join(embeddings_path, "images.pt")
        self.paths_file = os.path.join(embeddings_path, "paths.pkl")
        self.split_file = os.path.join(embeddings_path, "split_info.pt")
        
        # Check if files exist
        required_files = [self.metadata_path, self.embeddings_file, self.images_file, self.paths_file]
        if split != 'all':
            required_files.append(self.split_file)
            
        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # Load metadata
        self.metadata = torch.load(self.metadata_path)
        self.num_samples = self.metadata['num_samples']
        self.embedding_dim = self.metadata['embedding_dim']
        self.image_shape = self.metadata['image_shape']
        
        # Load paths
        import pickle
        with open(self.paths_file, 'rb') as f:
            self.paths = pickle.load(f)
        
        # Set up indices for split
        self.indices = None
        if split != 'all' and os.path.exists(self.split_file):
            split_info = torch.load(self.split_file)
            if split == 'train':
                self.indices = split_info['train_indices']
                logger.info(f"Loaded {len(self.indices)} training indices from {self.split_file}")
            elif split == 'val':
                self.indices = split_info['val_indices']
                logger.info(f"Loaded {len(self.indices)} validation indices from {self.split_file}")
            
        # Cache for chunks - using LRU cache
        from collections import OrderedDict
        self.cache = OrderedDict()
        
    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return self.num_samples
    
    def _get_chunk_idx(self, idx):
        # If using split, map the idx to the global index first
        if self.indices is not None:
            idx = self.indices[idx]
        return idx // self.chunk_size
    
    def _get_local_idx(self, idx):
        # If using split, map the idx to the global index first
        if self.indices is not None:
            idx = self.indices[idx]
        return idx % self.chunk_size
    
    def _load_chunk(self, chunk_idx):
        # Check if chunk is already in cache
        if chunk_idx in self.cache:
            # Move to the end of LRU
            self.cache.move_to_end(chunk_idx)
            return self.cache[chunk_idx]
        
        # Free memory if cache is too large
        if len(self.cache) >= self.max_cache_chunks:
            # Remove least recently used chunk (first item)
            self.cache.popitem(last=False)
            # Force garbage collection to free memory
            import gc
            gc.collect()
        
        # Load chunk
        start_idx = chunk_idx * self.chunk_size
        end_idx = min((chunk_idx + 1) * self.chunk_size, self.num_samples)
        
        try:
            # Memory-mapped tensors with error handling
            embeddings = torch.load(self.embeddings_file, map_location='cpu')[start_idx:end_idx]
            images = torch.load(self.images_file, map_location='cpu')[start_idx:end_idx]
            
            # Store in cache
            self.cache[chunk_idx] = (embeddings, images)
            logger.debug(f"Loaded chunk {chunk_idx} with {end_idx - start_idx} samples")
            
            return embeddings, images
        except Exception as e:
            logger.error(f"Error loading chunk {chunk_idx}: {e}")
            raise
    
    def __getitem__(self, idx):
        try:
            chunk_idx = self._get_chunk_idx(idx)
            local_idx = self._get_local_idx(idx)
            
            embeddings, images = self._load_chunk(chunk_idx)
            
            # Get the data
            embedding = embeddings[local_idx]
            image = images[local_idx]
            
            # Get the global index for path retrieval
            global_idx = idx if self.indices is None else self.indices[idx]
            
            if self.paths is not None:
                return embedding, image, self.paths[global_idx]
            else:
                return embedding, image
        except Exception as e:
            logger.error(f"Error getting item {idx}: {e}")
            raise


def extract_embeddings(encoder, dataloader, device="cuda", save_path=None, batch_process=True, max_retries=DEFAULT_CONFIG['data']['max_retries']):
    """
    Extract V-JEPA 2 embeddings from images.
    
    Arguments:
        encoder: V-JEPA 2 encoder model
        dataloader: DataLoader for the image dataset
        device: Device to run the encoder on
        save_path: Path to save embeddings incrementally to disk
        batch_process: Whether to process in batches to save memory
        
    Returns:
        embeddings: Tensor of embeddings or path to saved embeddings
        images: Tensor of original images or path to saved images
        image_paths: List of image paths
    """
    # Simple in-memory processing
    embeddings = []
    images = []
    paths = []
    
    encoder.eval()
    encoder = encoder.to(device)
    
    with torch.no_grad():
        for batch, paths_batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Process batch with retry mechanism
            for attempt in range(max_retries):
                try:
                    # Move batch to device
                    batch = batch.to(device)
                    original_batch = batch.clone()  # Keep original for output
                    
                    # Prepare batch for encoder (add time dimension if needed)
                    if batch.dim() == 4:  # [B, C, H, W]
                        # For V-JEPA 2 trained with 2 frames, duplicate the frame
                        batch = batch.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # [B, C, 2, H, W]
                    
                    # Get embeddings from encoder
                    output = encoder(batch)
                    
                    # Handle different output formats
                    if output.dim() == 5:  # [B, N, T, D] - remove time dimension
                        output = output.squeeze(2)
                    elif output.dim() == 4 and output.shape[2] == 1:  # [B, N, 1, D]
                        output = output.squeeze(2)
                    
                    # Store results
                    embeddings.append(output.cpu())
                    images.append(original_batch.cpu())  # Use original 4D images
                    paths.extend(paths_batch)
                    break  # Success, break retry loop
                    
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and attempt < max_retries - 1:
                        # CUDA OOM error - clear cache and reduce batch if possible
                        logger.warning(f"CUDA OOM error, clearing cache and retrying (attempt {attempt+1}/{max_retries})")
                        torch.cuda.empty_cache()
                        
                        # Allow some time for memory to clear
                        time.sleep(1)
                        continue
                    elif attempt == max_retries - 1:
                        logger.error(f"Error processing batch after {max_retries} attempts: {e}")
                        logger.error(f"Skipping this batch and continuing")
                    else:
                        logger.warning(f"Attempt {attempt+1} failed: {e}, retrying...")
                        continue
                        
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Error processing batch with shape {batch.shape}: {e}")
                        logger.error(f"Batch paths: {paths_batch}")
                        logger.error(f"Original batch shape: {original_batch.shape}")
                    else:
                        logger.warning(f"Attempt {attempt+1} failed: {e}, retrying...")
                        continue
    
    # Concatenate results if we have any successful batches
    if embeddings:
        embeddings = torch.cat(embeddings, dim=0)
        images = torch.cat(images, dim=0)
        
        print(f"Final embeddings shape: {embeddings.shape}")
        print(f"Final images shape: {images.shape}")
        
        # Save to disk if path provided
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            # Save data files
            torch.save(embeddings, os.path.join(save_path, "embeddings.pt"))
            torch.save(images, os.path.join(save_path, "images.pt"))
            
            # Save metadata for streaming dataset
            metadata = {
                'num_samples': len(embeddings),
                'embedding_dim': embeddings.shape[-1],
                'image_shape': images.shape[1:],
                'creation_time': time.time()
            }
            torch.save(metadata, os.path.join(save_path, "metadata.pt"))
            
            # Save paths
            import pickle
            with open(os.path.join(save_path, "paths.pkl"), 'wb') as f:
                pickle.dump(paths, f)
                
            logger.info(f"Saved embeddings, images, and metadata to {save_path}")
    else:
        print("No valid images were processed!")
        embeddings = torch.tensor([])
        images = torch.tensor([])
    
    return embeddings, images, paths


def create_frame_decoder_datasets(
    libero_image_dir, 
    encoder, 
    device="cuda", 
    batch_size=32,
    num_workers=4,
    val_split=0.1,
    seed=42,
    save_path="/home/tsunami/vjepa2/libero/embeddings",
    batch_process=True,
    use_streaming=False,
    chunk_size=DEFAULT_CONFIG['data']['streaming']['chunk_size']
):
    """
    Create train and validation datasets for frame decoder training.
    
    Arguments:
        libero_image_dir: Directory containing Libero images
        encoder: V-JEPA 2 encoder model
        device: Device to run the encoder on
        batch_size: Batch size for embedding extraction
        num_workers: Number of workers for data loading
        val_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        save_path: Path to save extracted embeddings
        batch_process: Whether to process in batches to save memory
        use_streaming: Whether to use streaming dataset (memory-efficient)
        chunk_size: Number of samples to load at once when using streaming
        
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataset and dataloader for Libero images
    transform = get_transform()
    
    # Check for existing embeddings
    embedding_file = os.path.join(save_path, "embeddings.pt")
    images_file = os.path.join(save_path, "images.pt")
    paths_file = os.path.join(save_path, "paths.pkl")
    metadata_file = os.path.join(save_path, "metadata.pt")
    
    if os.path.exists(embedding_file) and os.path.exists(images_file) and os.path.exists(paths_file) and os.path.exists(metadata_file):
        print(f"Loading pre-extracted embeddings from {save_path}")
        import pickle
        embeddings = torch.load(embedding_file)
        images = torch.load(images_file)
        with open(paths_file, 'rb') as f:
            paths = pickle.load(f)
    else:
        # Create dataset with improved filtering for empty/corrupted files
        import glob
        all_image_paths = sorted(glob.glob(os.path.join(libero_image_dir, "*.jpg")))
        valid_image_paths = []
        
        print(f"Found {len(all_image_paths)} total images, checking for valid files...")
        
        for path in tqdm(all_image_paths, desc="Filtering images"):
            try:
                # Check file size (should be > 0)
                if os.path.getsize(path) == 0:
                    continue
                    
                # Try to open the image to check if it's valid
                with Image.open(path) as img:
                    # Check if image has valid dimensions
                    if img.size[0] > 0 and img.size[1] > 0:
                        # Try to convert to RGB to ensure it's a valid image
                        img.convert('RGB')
                        valid_image_paths.append(path)
                        
            except (OSError, IOError, Exception) as e:
                # Skip corrupted or unreadable images
                continue
        
        print(f"Found {len(valid_image_paths)} valid images out of {len(all_image_paths)}")
        
        if len(valid_image_paths) == 0:
            raise ValueError(f"No valid images found in {libero_image_dir}")
        
        # Create specialized dataset with only valid images
        class FilteredLiberoImageDataset(Dataset):
            def __init__(self, valid_paths, transform=None):
                self.image_paths = valid_paths
                self.transform = transform
                
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
                max_retries = DEFAULT_CONFIG['data']['max_retries']
                
                for attempt in range(max_retries):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        
                        if self.transform:
                            image = self.transform(image)
                            
                        return image, img_path
                    except Exception as e:
                        if attempt == max_retries - 1:
                            logger.error(f"Error loading image {img_path} after {max_retries} attempts: {e}")
                            # Instead of returning a blank image, raise an exception so the caller can handle it properly
                            raise ValueError(f"Failed to load image {img_path} after {max_retries} attempts") from e
                        logger.warning(f"Attempt {attempt + 1} failed for {img_path}: {e}, retrying...")
        
        image_dataset = FilteredLiberoImageDataset(valid_image_paths, transform=transform)
        dataloader = DataLoader(
            image_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Extract embeddings
        print(f"Extracting embeddings from {len(image_dataset)} valid images...")
        embeddings, images, paths = extract_embeddings(encoder, dataloader, device, save_path, batch_process)
    
    # Check if we have valid data
    if len(embeddings) == 0:
        raise ValueError("No valid embeddings were extracted!")
    
    # Split into train and validation sets
    num_samples = len(embeddings)
    indices = torch.randperm(num_samples)
    val_size = int(val_split * num_samples)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    # Create datasets based on the streaming option
    if use_streaming:
        logger.info(f"Using memory-efficient streaming datasets with chunk size {chunk_size}")
        
        # We need to save the train/val split information for streaming datasets
        metadata = {
            'train_indices': train_indices.tolist(),
            'val_indices': val_indices.tolist(),
            'split_seed': seed,
            'val_split': val_split
        }
        
        # Save split information
        split_file = os.path.join(save_path, "split_info.pt")
        torch.save(metadata, split_file)
        
        # Create streaming datasets with appropriate split
        train_dataset = StreamingFrameDecoderDataset(
            embeddings_path=save_path, 
            chunk_size=chunk_size, 
            split='train'
        )
        
        val_dataset = StreamingFrameDecoderDataset(
            embeddings_path=save_path, 
            chunk_size=chunk_size, 
            split='val'
        )
    else:
        # Create standard in-memory datasets
        train_dataset = FrameDecoderDataset(
            embeddings[train_indices],
            images[train_indices],
            [paths[i] for i in train_indices]
        )
        
        val_dataset = FrameDecoderDataset(
            embeddings[val_indices],
            images[val_indices],
            [paths[i] for i in val_indices]
        )
    
    logger.info(f"Created datasets with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test LiberoImageDataset
    image_dir = "/home/tsunami/vjepa2/libero/libero_images"
    dataset = LiberoImageDataset(image_dir)
    print(f"Libero dataset contains {len(dataset)} images")
    
    # Test image loading
    image, path = dataset[0]
    print(f"Loaded image with shape {image.shape} from {path}")