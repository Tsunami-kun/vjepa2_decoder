# V-JEPA 2 Frame Decoder

A frame decoder for V-JEPA 2 latents that visualizes model predictions in pixel space. This feedforward network is trained on the Libero dataset using mean-squared error pixel reconstruction loss, following Sec. B.3 of the V-JEPA 2 tech report ([paper link](https://arxiv.org/abs/2506.09985)).

## Installation

Requires Python 3.8+:

```bash
git clone https://github.com/username/vjepa2-decoder.git
cd vjepa2-decoder
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from vjepa2_decoder import create_frame_decoder, load_vjepa2_models

# Load pre-trained models
encoder, predictor = load_vjepa2_models('path/to/vjepa2_checkpoint.pt')
decoder = create_frame_decoder('path/to/decoder_checkpoint.pt')

# Generate predictions
predictions = decoder(encoder(images))
```

## Training

Train on the Libero dataset:

```bash
python -m vjepa2_decoder.main train \
    --libero-dir /path/to/libero/images \
    --vjepa-checkpoint /path/to/vjepa2_checkpoint.pt \
    --model-type vit_large \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-4 \
    --use-streaming
```

## Prediction

Generate predictions with a trained decoder:

```bash
python -m vjepa2_decoder.main predict \
    --vjepa-checkpoint /path/to/vjepa2_checkpoint.pt \
    --decoder-checkpoint /path/to/decoder_checkpoint.pt \
    --input-dir /path/to/input/frames \
    --num-rollouts 5 \
    --output-dir ./predictions
```

## Configuration

Use custom configuration files:

```bash
python -m vjepa2_decoder.main train \
    --config /path/to/config.json \
    --libero-dir /path/to/libero/images \
    --vjepa-checkpoint /path/to/vjepa2_checkpoint.pt
```

Example config:

```json
{
  "model": {
    "patch_size": 16,
    "image_size": 224
  },
  "training": {
    "batch_size": 128,
    "learning_rate": 2e-4,
    "weight_decay": 0.02,
    "epochs": 100
  },
  "data": {
    "streaming": {
      "enabled": true,
      "chunk_size": 2000
    }
  }
}
```

## Architecture

The decoder uses progressive upsampling:

1. Input: V-JEPA 2 patch embeddings (B×N×D)
2. Reshape to 2D grid (B×14×14×D for 224×224 images with 16×16 patches)
3. Progressive upsampling:
   - 14×14×D → 28×28×512
   - 28×28×512 → 56×56×256
   - 56×56×256 → 112×112×128
   - 112×112×128 → 224×224×64
   - 224×224×64 → 224×224×3
4. Output: 224×224×3 RGB image

Each upsampling block contains:
- Transpose convolution for upsampling
- Group normalization + GELU activation
- Regular convolution for feature refinement
- Group normalization + GELU activation
- Dropout regularization

### Memory-Efficient Training

The `StreamingFrameDecoderDataset` enables training on large datasets by:

- Storing embeddings and images on disk
- Loading data in chunks during training
- Maintaining LRU cache of recent chunks
- Automatic memory management when cache limits are reached

## File Structure

```
vjepa2_decoder/
├── model.py          # Frame decoder architecture
├── dataset.py        # Dataset handling for Libero images and embeddings
├── train.py          # Training loop and optimization
├── predict.py        # Prediction and visualization pipeline
├── utils.py          # Utility functions
├── config.py         # Configuration management
├── main.py           # Command-line interface
└── __init__.py       # Package initialization
```

## License

MIT