"""
Tree Canopy Detection - Google Colab Setup & Inference Guide
Complete step-by-step notebook for running U-Net and Mask R-CNN on drone images
"""

# ============================================================================
# STEP 1: INSTALL DEPENDENCIES
# ============================================================================
# Run this in a Colab cell

!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install tensorflow keras
!pip install opencv-python numpy matplotlib scikit-image pillow tqdm albumentations
!pip install segmentation-models rasterio
!pip install git+https://github.com/matterport/Mask_RCNN.git

# ============================================================================
# STEP 2: CLONE REPOSITORY
# ============================================================================

!git clone https://github.com/miraz53/tree-canopy-detection.git
%cd tree-canopy-detection

# ============================================================================
# STEP 3: UPLOAD YOUR DRONE IMAGE
# ============================================================================

from google.colab import files
import os

print("Upload your drone image (TIFF or JPG)...")
uploaded = files.upload()

# Get the filename
image_name = list(uploaded.keys())[0]
print(f"✓ Uploaded: {image_name}")
print(f"  Size: {uploaded[image_name] / (1024**2):.2f} MB")

# Move to data folder
os.makedirs('data/raw', exist_ok=True)
os.rename(image_name, f'data/raw/{image_name}')
print(f"✓ Saved to: data/raw/{image_name}")

# ============================================================================
# STEP 4: EXPLORE YOUR IMAGE
# ============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_path = f'data/raw/{image_name}'

# Load image
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get image info
height, width = img_rgb.shape[:2]
print(f"\n📊 Image Information:")
print(f"  Resolution: {width} × {height} pixels")
print(f"  Channels: {img_rgb.shape[2]}")
print(f"  File size: {os.path.getsize(image_path) / (1024**2):.2f} MB")

# Display image sample
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Full image (downsampled for display)
display_img = cv2.resize(img_rgb, (1024, 1024))
axes[0].imshow(display_img)
axes[0].set_title('Drone Image (downsampled)')
axes[0].axis('off')

# Image statistics
axes[1].hist(img_rgb.reshape(-1, 3), bins=256, color=['r', 'g', 'b'], alpha=0.6)
axes[1].set_title('Color Distribution')
axes[1].set_xlabel('Pixel Value')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 5A: QUICK TREE CANOPY DETECTION (Color-based - No Training)
# ============================================================================
# Fastest method - no model training needed!

print("\n🌳 Running Quick Canopy Detection (Color-based)...")

# Convert to HSV for better green detection
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define green vegetation range
lower_green = np.array([35, 40, 40])
upper_green = np.array([90, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower_green, upper_green)

# Clean up with morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Calculate canopy coverage
canopy_pixels = cv2.countNonZero(mask)
total_pixels = mask.shape[0] * mask.shape[1]
canopy_percentage = (canopy_pixels / total_pixels) * 100

print(f"✓ Canopy Coverage: {canopy_percentage:.2f}%")

# Save results
os.makedirs('results', exist_ok=True)
cv2.imwrite('results/canopy_mask_quick.png', mask)
print(f"✓ Mask saved to: results/canopy_mask_quick.png")

# Display results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Original (downsampled)
axes[0].imshow(cv2.cvtColor(cv2.resize(img, (1024, 1024)), cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Drone Image')
axes[0].axis('off')

# Mask
mask_display = cv2.resize(mask, (1024, 1024))
axes[1].imshow(mask_display, cmap='gray')
axes[1].set_title(f'Detected Canopy ({canopy_percentage:.1f}%)')
axes[1].axis('off')

# Overlay
result = cv2.bitwise_and(img, img, mask=mask)
result_rgb = cv2.cvtColor(cv2.resize(result, (1024, 1024)), cv2.COLOR_BGR2RGB)
axes[2].imshow(result_rgb)
axes[2].set_title('Canopy Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 5B: ADVANCED - U-NET SEMANTIC SEGMENTATION
# ============================================================================
# Requires pre-trained model or training data
# Uncomment to use after getting training data

print("\n🧠 U-Net Semantic Segmentation Setup...")

try:
    import segmentation_models as smp
    import torch
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    
    if device.type == "cuda":
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create U-Net model
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    CLASSES = ['canopy']
    ACTIVATION = 'sigmoid'
    
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    )
    
    model = model.to(device)
    print(f"✓ U-Net model created successfully")
    print(f"  Encoder: {ENCODER}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("Note: Full U-Net training requires annotated training data.")

# ============================================================================
# STEP 5C: PREPROCESSING FOR LARGE IMAGES
# ============================================================================

print("\n🔧 Image Tiling for Large Images...")

def tile_image(image_path, tile_size=512, overlap=50):
    """Tile large image into overlapping chunks"""
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    tiles = []
    positions = []
    
    step = tile_size - overlap
    
    for y in range(0, height - tile_size + 1, step):
        for x in range(0, width - tile_size + 1, step):
            tile = img[y:y+tile_size, x:x+tile_size]
            tiles.append(tile)
            positions.append((x, y))
    
    return tiles, positions, (height, width)

# Tile the image
tile_size = 512
tiles, positions, original_size = tile_image(image_path, tile_size=tile_size)

print(f"✓ Image tiled into {len(tiles)} tiles")
print(f"  Tile size: {tile_size} × {tile_size}")
print(f"  Original size: {original_size[1]} × {original_size[0]}")

# Show a few tiles
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
for idx, ax in enumerate(axes.flat):
    if idx < len(tiles):
        tile_rgb = cv2.cvtColor(tiles[idx], cv2.COLOR_BGR2RGB)
        ax.imshow(tile_rgb)
        ax.set_title(f"Tile {idx+1} at {positions[idx]}")
        ax.axis('off')

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 6: BATCH PROCESS TILES WITH COLOR-BASED METHOD
# ============================================================================

print("\n⚙️  Processing all tiles...")

all_masks = []

for idx, tile in enumerate(tiles):
    # Convert to HSV
    hsv_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
    
    # Detect green
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    tile_mask = cv2.inRange(hsv_tile, lower_green, upper_green)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tile_mask = cv2.morphologyEx(tile_mask, cv2.MORPH_CLOSE, kernel)
    tile_mask = cv2.morphologyEx(tile_mask, cv2.MORPH_OPEN, kernel)
    
    all_masks.append(tile_mask)
    
    if (idx + 1) % 10 == 0:
        print(f"  ✓ Processed {idx + 1}/{len(tiles)} tiles")

print(f"✓ All tiles processed!")

# ============================================================================
# STEP 7: STITCH TILES BACK TOGETHER
# ============================================================================

def stitch_masks(masks, positions, original_size, tile_size=512, overlap=50):
    """Reconstruct full mask from tiles"""
    height, width = original_size
    final_mask = np.zeros((height, width), dtype=np.uint8)
    count_map = np.zeros((height, width), dtype=np.float32)
    
    for mask, (x, y) in zip(masks, positions):
        h, w = mask.shape
        final_mask[y:y+h, x:x+w] += mask
        count_map[y:y+h, x:x+w] += 1
    
    # Average overlapping regions
    count_map[count_map == 0] = 1  # Avoid division by zero
    final_mask = (final_mask / count_map).astype(np.uint8)
    
    return final_mask

print("🔨 Stitching tiles back together...")

final_mask = stitch_masks(all_masks, positions, original_size, tile_size=tile_size)

# Calculate final canopy coverage
canopy_pixels_final = cv2.countNonZero(final_mask)
total_pixels_final = final_mask.shape[0] * final_mask.shape[1]
canopy_percentage_final = (canopy_pixels_final / total_pixels_final) * 100

print(f"✓ Full-resolution mask created!")
print(f"  Canopy Coverage: {canopy_percentage_final:.2f}%")

# Save final mask
cv2.imwrite('results/canopy_mask_full.png', final_mask)
print(f"✓ Full mask saved to: results/canopy_mask_full.png")

# ============================================================================
# STEP 8: VISUALIZE FINAL RESULTS
# ============================================================================

print("\n📊 Final Results...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Original (downsampled for display)
img_small = cv2.resize(img_rgb, (1024, 1024))
axes[0].imshow(img_small)
axes[0].set_title('Original Drone Image', fontsize=14)
axes[0].axis('off')

# Full mask (downsampled)
mask_small = cv2.resize(final_mask, (1024, 1024))
axes[1].imshow(mask_small, cmap='hot')
axes[1].set_title(f'Tree Canopy Detection\n({canopy_percentage_final:.1f}% coverage)', fontsize=14)
axes[1].axis('off')

# Overlay
result_overlay = img_small.copy()
result_overlay[mask_small > 128] = [0, 255, 0]  # Green overlay
axes[2].imshow(cv2.addWeighted(img_small, 0.7, result_overlay, 0.3, 0))
axes[2].set_title('Canopy Overlay', fontsize=14)
axes[2].axis('off')

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 9: DOWNLOAD RESULTS
# ============================================================================

print("\n💾 Downloading results...")

from google.colab import files

# Zip results
import zipfile
import shutil

os.makedirs('results', exist_ok=True)

# Create zip file
with zipfile.ZipFile('tree_canopy_results.zip', 'w') as zipf:
    zipf.write('results/canopy_mask_quick.png')
    zipf.write('results/canopy_mask_full.png')

files.download('tree_canopy_results.zip')
print("✓ Results downloaded as tree_canopy_results.zip")

# ============================================================================
# STEP 10: NEXT STEPS - ADVANCED TRAINING
# ============================================================================

print("\n📚 To train U-Net or Mask R-CNN models, you need:")
print("  1. Annotated training dataset (images + masks)")
print("  2. Run training script: python models/unet/train.py")
print("  3. Use trained weights for inference")
print("\nFor more info, see: README.md in the repository")

print("\n✅ COMPLETE! Drone image canopy detection finished.")
