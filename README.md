# Tree Canopy Detection from Drone RGB Images

Deep learning models for automated tree canopy detection and segmentation using drone-captured RGB imagery.

## Features

- **U-Net Model**: Semantic segmentation for pixel-level canopy classification
- **Mask R-CNN Model**: Instance segmentation for individual tree detection and boundary delineation
- **Data Preprocessing**: Utilities for handling large high-resolution drone images
- **Training Pipeline**: Complete training and validation framework
- **Inference Scripts**: Ready-to-use inference on drone images

## Models

### 1. U-Net (Semantic Segmentation)
- Encoder-decoder architecture
- Best for: Fast canopy/non-canopy pixel classification
- Output: Binary mask of canopy areas
- Advantages: Fast, memory-efficient, good for large images

### 2. Mask R-CNN (Instance Segmentation)
- Region-based CNN
- Best for: Individual tree detection and separation
- Output: Bounding boxes + masks for each tree
- Advantages: Identifies individual trees, handles overlapping canopies

## Project Structure

```
tree-canopy-detection/
├── data/
│   ├── raw/              # Raw drone images
│   ├── processed/        # Preprocessed images
│   └── annotations/      # Ground truth masks
├── models/
│   ├── unet/
│   │   ├── model.py      # U-Net architecture
│   │   ├── train.py      # Training script
│   │   └── inference.py  # Inference script
│   └── maskrcnn/
│       ├── model.py      # Mask R-CNN architecture
│       ├── train.py      # Training script
│       └── inference.py  # Inference script
├── utils/
│   ├── data_loader.py    # Data loading utilities
│   ├── preprocessing.py  # Image preprocessing
│   └── metrics.py        # Evaluation metrics
├── requirements.txt      # Project dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/miraz53/tree-canopy-detection.git
cd tree-canopy-detection
pip install -r requirements.txt
```

## Quick Start

### U-Net Training

```python
from models.unet.train import train_unet

train_unet(
    data_dir='data/processed',
    epochs=50,
    batch_size=8,
    learning_rate=0.001
)
```

### Mask R-CNN Training

```python
from models.maskrcnn.train import train_maskrcnn

train_maskrcnn(
    data_dir='data/processed',
    epochs=50,
    batch_size=4,
    learning_rate=0.001
)
```

### Inference

```python
from models.unet.inference import predict_canopy

# U-Net prediction
mask = predict_canopy('path/to/drone_image.tif', model_path='weights/unet.pth')

# Mask R-CNN prediction
results = predict_trees('path/to/drone_image.tif', model_path='weights/maskrcnn.pth')
```

## Data Requirements

### Training Data Format
- **Images**: RGB TIFF or JPG (any resolution, will be tiled)
- **Annotations**: Binary masks (PNG, same dimensions as images)
  - White (255): Tree canopy
  - Black (0): Non-canopy

### Recommended Dataset Size
- Minimum: 100 annotated drone images
- Optimal: 500+ images for robust models

## Usage with Your 16k Image

For large drone images (16001 × 16000 pixels):

```python
from utils.preprocessing import tile_large_image
from models.unet.inference import predict_canopy

# Tile the large image
tiles = tile_large_image('drone_image.tif', tile_size=512)

# Process each tile
results = []
for tile in tiles:
    mask = predict_canopy(tile, model_path='weights/unet.pth')
    results.append(mask)

# Stitch results back together
final_mask = stitch_tiles(results, original_size=(16001, 16000))
```

## Metrics

- **IoU (Intersection over Union)**: Semantic segmentation accuracy
- **Dice Score**: F1-like metric for segmentation
- **AP (Average Precision)**: Instance segmentation accuracy (Mask R-CNN)
- **Mean Average Recall**: Detection recall across IoU thresholds

## References

- U-Net: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- Mask R-CNN: [He et al., 2017](https://arxiv.org/abs/1703.06870)

## License

MIT

## Contributing

Contributions welcome! Please submit PRs for improvements or bug fixes.