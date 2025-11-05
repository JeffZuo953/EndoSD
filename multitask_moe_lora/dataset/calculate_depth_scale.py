#!/usr/bin/env python3
"""
Calculate correct depth_scale for SCARED dataset
"""

import cv2
import numpy as np

# Read the depth image
depth_path = "/data/ziyi/multitask/data/SCARED/train/dataset_1/keyframe_1/data/depthmap_rectified/000008.png"
img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
valid = img[img > 0]

print("Raw PNG depth values:")
print(f"  dtype: {img.dtype}")
print(f"  min: {valid.min()}")
print(f"  max: {valid.max()}")
print(f"  mean: {valid.mean():.2f}")
print()

# Your reference data from the working example
print("Expected depth values (from your npz reference):")
print(f"  min: 0.03741455078125 m")
print(f"  max: 0.272216796875 m")
print(f"  mean: 0.12451171875 m")
print()

# Calculate required depth_scale
print("Calculating required depth_scale:")
print("-" * 80)

# Method 1: Based on max value
scale_from_max = valid.max() / 0.272216796875
print(f"Based on max value: {valid.max()} / 0.272 = {scale_from_max:.1f}")

# Method 2: Based on mean value
scale_from_mean = valid.mean() / 0.12451171875
print(f"Based on mean value: {valid.mean():.2f} / 0.1245 = {scale_from_mean:.1f}")

# Method 3: Based on min value
scale_from_min = valid.min() / 0.03741455078125
print(f"Based on min value: {valid.min()} / 0.0374 = {scale_from_min:.1f}")

print()
print("=" * 80)
print("RECOMMENDATION:")
print("=" * 80)

# Use mean as the most reliable
recommended_scale = round(scale_from_mean, -2)  # Round to nearest 100
print(f"Use depth_scale = {recommended_scale:.1f}")
print()

# Verify
print("Verification with recommended scale:")
depths_m = valid.astype(np.float32) / recommended_scale
print(f"  min: {depths_m.min():.6f} m")
print(f"  max: {depths_m.max():.6f} m")
print(f"  mean: {depths_m.mean():.6f} m")
print()

if depths_m.max() <= 0.3:
    print("✓ All values fit within max_depth=0.3")
else:
    print(f"✗ Max value {depths_m.max():.3f} exceeds max_depth=0.3")
    print(f"  Consider using max_depth={depths_m.max():.3f} or higher")
