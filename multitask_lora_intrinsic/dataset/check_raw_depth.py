#!/usr/bin/env python3
"""
Check raw depth PNG values to determine correct depth_scale
"""

import cv2
import numpy as np
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python check_raw_depth.py <path_to_depth_png>")
    print("\nExample:")
    print("  python check_raw_depth.py /data/ziyi/multitask/data/SCARED/train/dataset_1/keyframe_1/data/depthmap_rectified/000008.png")
    sys.exit(1)

depth_path = sys.argv[1]

if not os.path.exists(depth_path):
    print(f"Error: File not found: {depth_path}")
    sys.exit(1)

print(f"Loading: {depth_path}")
print()

# Read as 16-bit
depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

if depth_png is None:
    print("Error: Failed to read depth image")
    sys.exit(1)

print(f"Image shape: {depth_png.shape}")
print(f"Image dtype: {depth_png.dtype}")
print()

# Get valid depth values (non-zero)
valid_mask = depth_png > 0
valid_depths = depth_png[valid_mask]

print(f"Valid pixel count: {len(valid_depths)} / {depth_png.size} ({len(valid_depths)/depth_png.size*100:.2f}%)")
print()

print("Raw PNG values (16-bit integers):")
print(f"  Min: {np.min(valid_depths)}")
print(f"  Max: {np.max(valid_depths)}")
print(f"  Mean: {np.mean(valid_depths):.2f}")
print(f"  Median: {np.median(valid_depths):.2f}")
print(f"  Std: {np.std(valid_depths):.2f}")
print()

# Try different scale factors
print("Testing different depth_scale values:")
print("-" * 80)

scales = [1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0, 65535.0]
target_max = 0.3  # Expected max depth in meters

for scale in scales:
    depths_m = valid_depths.astype(np.float32) / scale
    min_d = np.min(depths_m)
    max_d = np.max(depths_m)
    mean_d = np.mean(depths_m)

    # Check if this scale makes sense
    if max_d <= target_max:
        status = "✓ POSSIBLE"
    else:
        status = "✗ Too large"

    print(f"  scale={scale:8.1f} -> min={min_d:.6f}m, max={max_d:.6f}m, mean={mean_d:.6f}m  {status}")

print()
print("=" * 80)
print("Recommendation:")
print("=" * 80)

# Find best scale
best_scale = None
for scale in [1000.0, 5000.0, 10000.0]:
    max_d = np.max(valid_depths.astype(np.float32) / scale)
    if max_d <= target_max:
        best_scale = scale
        break

if best_scale:
    print(f"Use depth_scale = {best_scale}")
    depths_m = valid_depths.astype(np.float32) / best_scale
    print(f"This gives depth range: [{np.min(depths_m):.6f}m, {np.max(depths_m):.6f}m]")
    print(f"Mean depth: {np.mean(depths_m):.6f}m")
else:
    # Calculate required scale
    max_raw = np.max(valid_depths)
    required_scale = max_raw / target_max
    print(f"Raw max value {max_raw} exceeds expected range.")
    print(f"Required scale would be: {required_scale:.1f}")
    print(f"Please verify the expected max_depth value (currently {target_max}m)")

print()
print("Sample of first 20 valid raw values:")
print(valid_depths[:20])
