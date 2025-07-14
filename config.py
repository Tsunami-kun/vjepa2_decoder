"""
Configuration settings for V-JEPA 2 frame decoder.
"""

import os
import json
import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration settings
DEFAULT_CONFIG = {
    'model': {
        'patch_size': 16,
        'image_size': 224,
        'num_frames': 2,
        'tubelet_size': 2,
        'channels': 3,
        'dropout_rate': 0.1,
        'embedding_dims': {
            'vit_large': 1024,
            'vit_huge': 1280,
            'vit_giant': 1408,
            'vit_giant_384': 1408,
        }
    },
    'training': {
        'batch_size': 64,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'epochs': 50,
        'val_split': 0.1,
        'seed': 42,
        'log_interval': 10,
        'optimizer': 'adamw',  # Options: 'adam', 'adamw', 'sgd'
        'scheduler': 'cosine',  # Options: 'cosine', 'step', 'plateau', 'none'
        'scheduler_params': {
            'cosine': {},
            'step': {'step_size': 10, 'gamma': 0.1},
            'plateau': {'patience': 5, 'factor': 0.5},
        },
        'early_stopping': {
            'enabled': False,
            'patience': 10,
            'min_delta': 0.0001
        },
    },
    'data': {
        'normalize_mean': [0.485, 0.456, 0.406],
        'normalize_std': [0.229, 0.224, 0.225],
        'max_retries': 3,
        'num_workers': 4,
        'pin_memory': True,
        'prefetch_factor': 2,
        'streaming': {
            'enabled': False,
            'chunk_size': 1000,
            'max_cache_chunks': 3
        },
    },
    'paths': {
        'default_output_dir': './runs/frame_decoder',
        'default_embeddings_dir': './embeddings',
        'default_checkpoint_dir': './checkpoints',
        'config_file': './frame_decoder_config.json',
    },
    'logging': {
        'level': 'INFO',  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
        'log_to_file': False,
        'log_file': './logs/frame_decoder.log',
    },
    'prediction': {
        'num_rollouts': 5,
        'fps': 5,
        'save_visualizations': True,
        'save_video': True,
    }
}


def get_config(overrides=None, config_file=None):
    """
    Get configuration with optional overrides from dictionary or config file.
    
    Args:
        overrides: Dictionary with values to override in the default config
        config_file: Path to JSON configuration file
        
    Returns:
        Modified configuration dictionary
    """
    import copy
    config = copy.deepcopy(DEFAULT_CONFIG)
    
    # First load from config file if provided
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            # Update config with file values
            _update_nested_dict(config, file_config)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading config from {config_file}: {e}")
    
    # Then apply direct overrides which take precedence
    if overrides:
        _update_nested_dict(config, overrides)
        
    # Set up logging based on config
    _configure_logging(config['logging'])
                
    return config


def _update_nested_dict(d, u):
    """
    Update nested dictionary recursively.
    
    Args:
        d: Dictionary to update
        u: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    import copy
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _update_nested_dict(d[k], v)
        else:
            d[k] = copy.deepcopy(v)
    return d


def _configure_logging(logging_config):
    """
    Configure logging based on config.
    
    Args:
        logging_config: Logging configuration dictionary
    """
    log_level = getattr(logging, logging_config.get('level', 'INFO'))
    
    # Set up root logger
    logging.getLogger().setLevel(log_level)
    
    # Add file handler if enabled
    if logging_config.get('log_to_file', False):
        log_file = logging_config.get('log_file', './logs/frame_decoder.log')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)


def save_config(config, file_path=None):
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        file_path: Path to save the config (defaults to config['paths']['config_file'])
        
    Returns:
        Path to the saved config file
    """
    if file_path is None:
        file_path = config['paths'].get('config_file', './frame_decoder_config.json')
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save config
    try:
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Configuration saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving config to {file_path}: {e}")
        return None


def get_model_dim(model_type):
    """
    Get the embedding dimension for a specific model type.
    
    Args:
        model_type: V-JEPA 2 model type (e.g., 'vit_large')
        
    Returns:
        Embedding dimension for the specified model type
    
    Raises:
        ValueError: If model_type is not supported
    """
    embedding_dims = DEFAULT_CONFIG['model']['embedding_dims']
    if model_type not in embedding_dims:
        raise ValueError(f"Unsupported model type: {model_type}. "
                         f"Choose from {list(embedding_dims.keys())}.")
    return embedding_dims[model_type]


def get_transform(image_size=None, config=None, augment=False):
    """
    Get standard image transformation pipeline.
    
    Args:
        image_size: Optional override for image size
        config: Optional config dictionary
        augment: Whether to use data augmentation
        
    Returns:
        torchvision.transforms.Compose object
    """
    from torchvision import transforms
    
    # Use provided config or default
    if config is None:
        config = DEFAULT_CONFIG
        
    if image_size is None:
        image_size = config['model']['image_size']
    
    mean = config['data']['normalize_mean']
    std = config['data']['normalize_std']
    
    # Basic transforms for all cases
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    
    # Add augmentations for training
    if augment:
        # Insert augmentation transforms before ToTensor
        transform_list = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    
    return transforms.Compose(transform_list)


def get_optimizer(model_parameters, config=None):
    """
    Create optimizer based on configuration.
    
    Args:
        model_parameters: Model parameters to optimize
        config: Configuration dictionary
        
    Returns:
        PyTorch optimizer
    """
    # Use provided config or default
    if config is None:
        config = DEFAULT_CONFIG
        
    optimizer_type = config['training'].get('optimizer', 'adamw').lower()
    lr = config['training'].get('learning_rate', 1e-4)
    weight_decay = config['training'].get('weight_decay', 0.01)
    
    # Create optimizer based on type
    if optimizer_type == 'adam':
        return torch.optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = config['training'].get('momentum', 0.9)
        return torch.optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        logger.warning(f"Unknown optimizer type: {optimizer_type}, using AdamW")
        return torch.optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)


def get_scheduler(optimizer, config=None):
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Configuration dictionary
        
    Returns:
        PyTorch learning rate scheduler or None if disabled
    """
    # Use provided config or default
    if config is None:
        config = DEFAULT_CONFIG
        
    scheduler_type = config['training'].get('scheduler', 'cosine').lower()
    epochs = config['training'].get('epochs', 50)
    
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'step':
        step_params = config['training']['scheduler_params']['step']
        step_size = step_params.get('step_size', 10)
        gamma = step_params.get('gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'plateau':
        plateau_params = config['training']['scheduler_params']['plateau']
        patience = plateau_params.get('patience', 5)
        factor = plateau_params.get('factor', 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    elif scheduler_type == 'none':
        return None
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}, using None")
        return None