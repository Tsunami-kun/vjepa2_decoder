"""
Prediction visualization pipeline for V-JEPA 2 frame decoder.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from PIL import Image
import argparse
from einops import rearrange
from tqdm import tqdm
import torchvision.transforms as transforms
import glob
import cv2

from .config import get_transform, DEFAULT_CONFIG, get_config

# Local imports
from .model import create_frame_decoder


def load_frame_decoder(model_path, model_type="vit_large"):
    """
    Load a trained frame decoder model.
    
    Arguments:
        model_path: Path to the model checkpoint
        model_type: V-JEPA 2 model type
        
    Returns:
        Frame decoder model
    """
    model = create_frame_decoder(model_type=model_type)
    state_dict = torch.load(model_path, map_location="cpu")
    
    # Handle loading from full checkpoint or just state_dict
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
        
    return model


def load_vjepa2_models(checkpoint_path, device="cuda", num_frames=1):
    """
    Load V-JEPA 2 encoder and predictor.
    
    Arguments:
        checkpoint_path: Path to the V-JEPA 2 checkpoint
        device: Device to load the models onto
        
    Returns:
        Encoder and predictor models
    """
    import torch
    from src.hub.backbones import _clean_backbone_key
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load encoder
    from src.models.vision_transformer import vit_large
    # Use original num_frames from checkpoint (2 frames)
    original_num_frames = 2
    encoder = vit_large(
        patch_size=16,
        img_size=(224, 224),
        num_frames=original_num_frames,  # Use original configuration
        tubelet_size=2,
        use_sdpa=True,
        use_rope=True,
    )
    
    # Load predictor
    from src.models.predictor import vit_predictor
    predictor = vit_predictor(
        img_size=(224, 224),
        patch_size=16,
        use_mask_tokens=True,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=384,
        num_frames=16,
        tubelet_size=2,
        depth=12,
        num_heads=12,
        num_mask_tokens=10,
        use_rope=True,
        use_sdpa=True,
    )
    
    # Load state dicts
    encoder_state_dict = _clean_backbone_key(checkpoint["encoder"])
    encoder.load_state_dict(encoder_state_dict, strict=False)
    
    predictor_state_dict = _clean_backbone_key(checkpoint["predictor"])
    predictor.load_state_dict(predictor_state_dict, strict=False)
    
    # Move to device
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    
    # Set to evaluation mode
    encoder.eval()
    predictor.eval()
    
    return encoder, predictor


def preprocess_frames(frames, device="cuda"):
    """
    Preprocess frames for V-JEPA 2.
    
    Arguments:
        frames: List of frame tensors or PIL images
        device: Device to load the tensors onto
        
    Returns:
        Preprocessed tensor of shape (batch_size, channels, time, height, width)
    """
    # Convert PIL images to tensors if necessary
    if isinstance(frames[0], Image.Image):
        transform = get_transform()
        frames = [transform(frame) for frame in frames]
    
    # Stack frames and add batch dimension if necessary
    if isinstance(frames, list):
        frames = torch.stack(frames, dim=0)  # Stack along time dimension
    
    # Add batch dimension if not present
    if len(frames.shape) == 4:  # C, T, H, W
        frames = frames.unsqueeze(0)  # B, C, T, H, W
    
    # Move to device
    frames = frames.to(device)
    
    return frames


def create_masks_for_prediction(num_patches, target_indices, device="cuda"):
    """
    Create masks for target patches.
    
    Arguments:
        num_patches: Total number of patches
        target_indices: Indices of patches to mask
        device: Device for the mask tensor
        
    Returns:
        Binary mask tensor where 1 indicates patches to predict
    """
    mask = torch.zeros(num_patches, dtype=torch.bool, device=device)
    mask[target_indices] = True
    return mask


def generate_prediction_sequence(encoder, predictor, decoder, frames, num_rollouts=5, device="cuda"):
    """
    Generate a sequence of predictions using V-JEPA 2 and frame decoder.
    
    Arguments:
        encoder: V-JEPA 2 encoder model
        predictor: V-JEPA 2 predictor model
        decoder: Frame decoder model
        frames: Input frames (B, C, T, H, W)
        num_rollouts: Number of frames to predict ahead
        device: Device to run prediction on
        
    Returns:
        List of predicted frames and the encoded representations
    """
    encoder.eval()
    predictor.eval()
    decoder.eval()
    
    # Move models to device
    encoder = encoder.to(device)
    predictor = predictor.to(device)
    decoder = decoder.to(device)
    
    # Preprocess frames
    frames = preprocess_frames(frames, device)
    
    # Extract encoded representations for initial frames
    with torch.no_grad():
        encoded_frames = encoder(frames)
    
    # Determine patch dimensions
    patch_size = 16
    image_size = 224
    num_patches_per_dim = image_size // patch_size
    num_patches = num_patches_per_dim ** 2
    
    # Generate predictions
    predictions = []
    representations = []
    current_representation = encoded_frames
    
    # Generate the decoded frame for initial representation
    with torch.no_grad():
        initial_decoded = decoder(current_representation)
        predictions.append(initial_decoded)
        representations.append(current_representation.clone())
    
    # Perform rollout predictions
    for step in range(num_rollouts):
        with torch.no_grad():
            # Create mask for the predictor (predict next frame)
            masks = create_masks_for_prediction(num_patches, list(range(num_patches)), device)
            
            # Generate prediction
            predicted_representation = predictor(
                current_representation,
                masks=masks.unsqueeze(0),  # Add batch dimension
            )
            
            # Decode the predicted representation to pixels
            decoded_frame = decoder(predicted_representation)
            
            # Store results
            predictions.append(decoded_frame)
            representations.append(predicted_representation.clone())
            
            # Update current representation for next step
            current_representation = predicted_representation
    
    return predictions, representations


def visualize_prediction_sequence(predictions, output_path=None, show=True):
    """
    Visualize a sequence of predicted frames.
    
    Arguments:
        predictions: List of predicted frames
        output_path: Path to save the visualization
        show: Whether to display the visualization
        
    Returns:
        None
    """
    # Create a grid of images
    num_frames = len(predictions)
    batch_size = predictions[0].shape[0]
    
    # Create separate rows for each sample in the batch
    for b in range(batch_size):
        # Extract frames for this sample
        sample_frames = [pred[b] for pred in predictions]
        
        # Create grid
        grid = make_grid(sample_frames, nrow=num_frames, normalize=True)
        
        # Convert to numpy for visualization
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        # Plot
        plt.figure(figsize=(20, 4))
        plt.imshow(grid_np)
        plt.axis('off')
        
        # Add labels
        for i in range(num_frames):
            label = "Initial" if i == 0 else f"Pred {i}"
            plt.text(i * grid_np.shape[1] / num_frames + grid_np.shape[1] / (2 * num_frames),
                     grid_np.shape[0] - 10,
                     label,
                     ha='center',
                     color='white',
                     fontsize=14,
                     bbox=dict(facecolor='black', alpha=0.5))
        
        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if batch_size > 1:
                # Append sample index if multiple samples
                base, ext = os.path.splitext(output_path)
                path = f"{base}_sample_{b}{ext}"
            else:
                path = output_path
            plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
            print(f"Saved visualization to {path}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close()


def create_video_from_predictions(predictions, output_path, fps=5):
    """
    Create a video from a sequence of predicted frames.
    
    Arguments:
        predictions: List of predicted frames
        output_path: Path to save the video
        fps: Frames per second
        
    Returns:
        None
    """
    batch_size = predictions[0].shape[0]
    
    for b in range(batch_size):
        # Extract frames for this sample
        sample_frames = [pred[b] for pred in predictions]
        
        # Convert to numpy images
        frames = []
        for frame in sample_frames:
            # Convert from tensor to numpy
            img = frame.permute(1, 2, 0).cpu().numpy()
            # Normalize to [0, 255] and convert to uint8
            img = (img * 255).astype(np.uint8)
            # Convert from RGB to BGR for OpenCV
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            frames.append(img)
        
        # Create video path
        if batch_size > 1:
            # Append sample index if multiple samples
            base, ext = os.path.splitext(output_path)
            video_path = f"{base}_sample_{b}{ext}"
        else:
            video_path = output_path
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        # Create video writer
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        # Add frames to video
        for frame in frames:
            writer.write(frame)
        
        # Release writer
        writer.release()
        print(f"Saved video to {video_path}")


def main(args):
    """Main function for prediction visualization."""
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load frame decoder
    decoder = load_frame_decoder(
        model_path=args.decoder_checkpoint,
        model_type=args.model_type,
    )
    decoder.to(device)
    decoder.eval()
    
    # Load V-JEPA 2 models
    encoder, predictor = load_vjepa2_models(
        checkpoint_path=args.vjepa_checkpoint,
        device=device,
    )
    
    # Load frames
    frame_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")))[:args.num_input_frames]
    if len(frame_paths) < args.num_input_frames:
        print(f"Warning: Found {len(frame_paths)} frames, but requested {args.num_input_frames}")
    
    # Load and preprocess frames
    frames = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        frames.append(img)
    
    print(f"Loaded {len(frames)} frames")
    
    # Generate predictions
    predictions, representations = generate_prediction_sequence(
        encoder=encoder,
        predictor=predictor,
        decoder=decoder,
        frames=frames,
        num_rollouts=args.num_rollouts,
        device=device,
    )
    
    print(f"Generated {len(predictions)} frames (initial + {args.num_rollouts} rollouts)")
    
    # Visualize predictions
    visualize_prediction_sequence(
        predictions=predictions,
        output_path=os.path.join(args.output_dir, "prediction_sequence.png"),
        show=args.show,
    )
    
    # Create video
    create_video_from_predictions(
        predictions=predictions,
        output_path=os.path.join(args.output_dir, "prediction_video.mp4"),
        fps=args.fps,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize V-JEPA 2 predictions with frame decoder")
    parser.add_argument("--decoder-checkpoint", type=str, required=True,
                        help="Path to the frame decoder checkpoint")
    parser.add_argument("--vjepa-checkpoint", type=str, required=True,
                        help="Path to the V-JEPA 2 checkpoint")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory containing input frames")
    parser.add_argument("--output-dir", type=str, default="./output/predictions",
                        help="Directory to save predictions")
    parser.add_argument("--model-type", type=str, default="vit_large",
                        choices=["vit_large", "vit_huge", "vit_giant", "vit_giant_384"],
                        help="V-JEPA 2 model type")
    parser.add_argument("--num-input-frames", type=int, default=1,
                        help="Number of input frames")
    parser.add_argument("--num-rollouts", type=int, default=5,
                        help="Number of rollout steps")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run prediction on")
    parser.add_argument("--fps", type=int, default=5,
                        help="Frames per second for output video")
    parser.add_argument("--show", action="store_true",
                        help="Show visualization")
    
    args = parser.parse_args()
    
    main(args)