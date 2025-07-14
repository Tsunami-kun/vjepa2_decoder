#!/usr/bin/env python
"""
Main script for training and using the V-JEPA 2 frame decoder.
A unified interface for training and using the decoder.
"""

import os
import torch
import argparse
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame_decoder.model import create_frame_decoder
from frame_decoder.dataset import create_frame_decoder_datasets
from frame_decoder.train import train
from frame_decoder.predict import load_vjepa2_models, generate_prediction_sequence, visualize_prediction_sequence, create_video_from_predictions
from frame_decoder.utils import get_device, create_run_directory


def train_decoder(args):
    """Train a frame decoder model."""
    print("=== Training Frame Decoder ===")
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = args.output_dir or create_run_directory("./runs/frame_decoder")
    print(f"Saving outputs to {output_dir}")
    
    # Create embeddings directory if needed
    embeddings_dir = os.path.join(os.path.dirname(args.libero_dir), "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    print(f"Embeddings will be saved to {embeddings_dir}")
    
    # Load V-JEPA 2 encoder
    print(f"Loading V-JEPA 2 models from {args.vjepa_checkpoint}")
    encoder, _ = load_vjepa2_models(args.vjepa_checkpoint, device, num_frames=1)
    
    # Create datasets
    print(f"Creating datasets from {args.libero_dir}")
    train_dataset, val_dataset = create_frame_decoder_datasets(
        libero_image_dir=args.libero_dir,
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        save_path=embeddings_dir,
        batch_process=args.batch_process,
    )
    
    # Create model
    print(f"Creating frame decoder model ({args.model_type})")
    decoder = create_frame_decoder(
        model_type=args.model_type,
        patch_size=16,
        image_size=224,
    )
    
    # Train model
    print("Starting training...")
    train(
        model=decoder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        num_workers=args.num_workers,
    )
    
    print(f"Training complete! Model saved to {output_dir}")


def predict_with_decoder(args):
    """Generate predictions using a trained frame decoder."""
    print("=== Generating Predictions ===")
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = args.output_dir or create_run_directory("./runs/predictions")
    print(f"Saving outputs to {output_dir}")
    
    # Load V-JEPA 2 models
    print(f"Loading V-JEPA 2 models from {args.vjepa_checkpoint}")
    encoder, predictor = load_vjepa2_models(args.vjepa_checkpoint, device, num_frames=1)
    
    # Load frame decoder
    print(f"Loading frame decoder from {args.decoder_checkpoint}")
    decoder = create_frame_decoder(args.model_type)
    
    # Load state dict
    state_dict = torch.load(args.decoder_checkpoint, map_location="cpu")
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        decoder.load_state_dict(state_dict["model_state_dict"])
    else:
        decoder.load_state_dict(state_dict)
    decoder = decoder.to(device)
    
    # Load frames
    from PIL import Image
    import glob
    print(f"Loading frames from {args.input_dir}")
    frame_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.jpg")))[:args.num_input_frames]
    if not frame_paths:
        print(f"No frames found in {args.input_dir}")
        return
    
    print(f"Found {len(frame_paths)} frames")
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    
    # Generate predictions
    print(f"Generating {args.num_rollouts} prediction rollouts...")
    predictions, _ = generate_prediction_sequence(
        encoder=encoder,
        predictor=predictor,
        decoder=decoder,
        frames=frames,
        num_rollouts=args.num_rollouts,
        device=device,
    )
    
    # Visualize predictions
    print("Creating visualizations...")
    visualize_prediction_sequence(
        predictions=predictions,
        output_path=os.path.join(output_dir, "prediction_sequence.png"),
        show=False,
    )
    
    # Create video
    create_video_from_predictions(
        predictions=predictions,
        output_path=os.path.join(output_dir, "prediction_video.mp4"),
        fps=args.fps,
    )
    
    print(f"Predictions saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="V-JEPA 2 Frame Decoder")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a frame decoder")
    train_parser.add_argument("--libero-dir", type=str, required=True,
                              help="Directory containing Libero images")
    train_parser.add_argument("--vjepa-checkpoint", type=str, required=True,
                              help="Path to V-JEPA 2 checkpoint")
    train_parser.add_argument("--output-dir", type=str, default=None,
                              help="Directory to save model outputs")
    train_parser.add_argument("--model-type", type=str, default="vit_large",
                              choices=["vit_large", "vit_huge", "vit_giant", "vit_giant_384"],
                              help="V-JEPA 2 model type")
    train_parser.add_argument("--epochs", type=int, default=50,
                              help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=64,
                              help="Batch size for training")
    train_parser.add_argument("--lr", type=float, default=1e-4,
                              help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=0.01,
                              help="Weight decay")
    train_parser.add_argument("--num-workers", type=int, default=4,
                              help="Number of data loading workers")
    train_parser.add_argument("--val-split", type=float, default=0.1,
                              help="Validation split ratio")
    train_parser.add_argument("--batch-process", action="store_true",
                              help="Process embeddings in batches to save memory")
    train_parser.add_argument("--seed", type=int, default=42,
                              help="Random seed")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Generate predictions with a trained decoder")
    predict_parser.add_argument("--vjepa-checkpoint", type=str, required=True,
                                help="Path to V-JEPA 2 checkpoint")
    predict_parser.add_argument("--decoder-checkpoint", type=str, required=True,
                                help="Path to trained frame decoder checkpoint")
    predict_parser.add_argument("--input-dir", type=str, required=True,
                                help="Directory containing input frames")
    predict_parser.add_argument("--output-dir", type=str, default=None,
                                help="Directory to save prediction outputs")
    predict_parser.add_argument("--model-type", type=str, default="vit_large",
                                choices=["vit_large", "vit_huge", "vit_giant", "vit_giant_384"],
                                help="V-JEPA 2 model type")
    predict_parser.add_argument("--num-input-frames", type=int, default=1,
                                help="Number of input frames")
    predict_parser.add_argument("--num-rollouts", type=int, default=5,
                                help="Number of rollout steps")
    predict_parser.add_argument("--fps", type=int, default=5,
                                help="Frames per second for output video")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_decoder(args)
    elif args.command == "predict":
        predict_with_decoder(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()