#!/usr/bin/env python3
"""
Visualize depth map and valid mask to understand why valid pixels are ~50%
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python visualize_depth_mask.py <path_to_pt_file>")
    print("\nExample:")
    print("  python visualize_depth_mask.py /data/ziyi/multitask/data/SCARED/cache/train/dataset_1/keyframe_1/data/left_rectified/000008.pt")
    sys.exit(1)

pt_path = sys.argv[1]
print(f"Loading: {pt_path}\n")

# Load cached data
data = torch.load(pt_path, map_location='cpu')

depth = data['depth'].numpy()
valid_mask = data['valid_mask'].numpy()
image = data['image'].numpy()

# Convert image from (C, H, W) to (H, W, C) and denormalize
image_vis = image.transpose(1, 2, 0)
# Denormalize (approximate, for visualization)
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
image_vis = image_vis * std + mean
image_vis = np.clip(image_vis, 0, 1)

print(f"Depth shape: {depth.shape}")
print(f"Valid pixels: {valid_mask.sum()} / {valid_mask.size} ({valid_mask.mean()*100:.2f}%)")
print(f"Invalid pixels: {(~valid_mask).sum()} / {valid_mask.size} ({(~valid_mask).mean()*100:.2f}%)")
print()

# Analyze spatial distribution of invalid pixels
invalid_mask = ~valid_mask
h, w = depth.shape

# Check edges
edge_size = 50
edges = {
    'top': invalid_mask[:edge_size, :].mean(),
    'bottom': invalid_mask[-edge_size:, :].mean(),
    'left': invalid_mask[:, :edge_size].mean(),
    'right': invalid_mask[:, -edge_size:].mean(),
    'center': invalid_mask[h//4:3*h//4, w//4:3*w//4].mean(),
}

print("Invalid pixel distribution:")
print(f"  Top edge ({edge_size}px): {edges['top']*100:.1f}%")
print(f"  Bottom edge ({edge_size}px): {edges['bottom']*100:.1f}%")
print(f"  Left edge ({edge_size}px): {edges['left']*100:.1f}%")
print(f"  Right edge ({edge_size}px): {edges['right']*100:.1f}%")
print(f"  Center region: {edges['center']*100:.1f}%")
print()

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: RGB, Depth, Valid Mask
axes[0, 0].imshow(image_vis)
axes[0, 0].set_title('RGB Image')
axes[0, 0].axis('off')

im1 = axes[0, 1].imshow(depth, cmap='jet', vmin=0, vmax=0.3)
axes[0, 1].set_title(f'Depth Map (max=0.3m)')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

axes[0, 2].imshow(valid_mask, cmap='gray')
axes[0, 2].set_title(f'Valid Mask ({valid_mask.mean()*100:.1f}% valid)')
axes[0, 2].axis('off')

# Row 2: Invalid regions overlay, depth histogram, spatial analysis
# Invalid regions in red overlay on RGB
axes[1, 0].imshow(image_vis)
invalid_overlay = np.zeros((*depth.shape, 4))
invalid_overlay[~valid_mask] = [1, 0, 0, 0.5]  # Red with 50% alpha
axes[1, 0].imshow(invalid_overlay)
axes[1, 0].set_title('Invalid Regions (Red Overlay)')
axes[1, 0].axis('off')

# Depth histogram (valid pixels only)
valid_depths = depth[valid_mask]
axes[1, 1].hist(valid_depths, bins=50, color='blue', alpha=0.7)
axes[1, 1].set_xlabel('Depth (m)')
axes[1, 1].set_ylabel('Pixel Count')
axes[1, 1].set_title(f'Valid Depth Distribution\n(all values = 0.3m)')
axes[1, 1].grid(True, alpha=0.3)

# Spatial distribution: row-wise and column-wise invalid ratio
row_invalid_ratio = (~valid_mask).mean(axis=1) * 100
col_invalid_ratio = (~valid_mask).mean(axis=0) * 100

ax_row = axes[1, 2]
ax_row.plot(row_invalid_ratio, range(len(row_invalid_ratio)), 'b-', linewidth=2)
ax_row.set_xlabel('Invalid %')
ax_row.set_ylabel('Row')
ax_row.set_title('Invalid Pixels by Row')
ax_row.invert_yaxis()
ax_row.grid(True, alpha=0.3)
ax_row.set_xlim([0, 100])

plt.tight_layout()

# Save figure
output_path = pt_path.replace('.pt', '_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Visualization saved to: {output_path}")
print("\nOpening plot...")
plt.show()

print("\nConclusion:")
print("-" * 80)
if edges['center'] > 0.3:
    print("✓ High invalid ratio even in center region")
    print("  → Likely due to texture-less surfaces or occlusions in the scene")
elif max(edges['top'], edges['bottom'], edges['left'], edges['right']) > 0.8:
    print("✓ Invalid pixels mainly at edges")
    print("  → Typical for stereo vision (matching window constraints)")
else:
    print("✓ Invalid pixels distributed throughout")
    print("  → Combination of edge effects and scene characteristics")
